import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, NamedTuple

# -----------------------------
# 1. PARAMETER DEFINITIONS
# -----------------------------

# Simulation controls
N_ITERATIONS = 5000
SIMULATION_DAYS = 365
OPERATING_HOURS_PER_DAY = 24.0

# Order-related
BASE_DAILY_ORDERS = 1000      # Mean daily orders
DAILY_ORDER_NORMAL_DIST_STD = 100  # Standard deviation for normally distributed daily order volumes
AVERAGE_ORDER_VALUE = 50.0

# Shipping parameters
SHIPPING_TIME_MU = 1.1        # Lognormal mu for ~3 day median delivery
SHIPPING_TIME_SIGMA = 0.3     # Controls spread of delivery times
DELAY_THRESHOLD = 5           # Orders taking longer than 5 days are considered delayed

# Delay costs (Progressive scale based on total shipping time)
DELAY_COSTS = {
    5: 10.0,   # $10 compensation for 5-7 day delivery
    7: 20.0,   # $20 compensation for 7-10 day delivery
    10: 30.0   # $30 compensation for 10+ day delivery
}

# Defect rate
BASE_DEFECT_RATE = 0.02
DEFECT_CLUSTER_PROBABILITY = 0.01  # Probability of defect cluster
DEFECT_CLUSTER_MULTIPLIER = 5      # Increased defect rate during cluster
COST_PER_DEFECT = AVERAGE_ORDER_VALUE * 1.2   # Direct cost per defect

# Spoilage parameters
BASE_SPOILAGE_THRESHOLD = 6.0     # Days until item is considered spoiled
SEASONAL_SPOILAGE_IMPACT = 0.2    # Seasonal variation factor
COST_PER_SPOILED = AVERAGE_ORDER_VALUE * 1.2
CURRENT_DAY = 0                   # Global counter for seasonal effects

# Warehouse parameters
NUM_WAREHOUSES = 5
WAREHOUSE_THROUGHPUT_SHARES = [1.0 / NUM_WAREHOUSES] * NUM_WAREHOUSES
MTBO_MEAN = 60.0                   # Mean Time Between Outages (days)
OUTAGE_DURATION_MEAN = 4.0         # Mean outage duration (hours)
OUTAGE_COST_PER_HOUR = 4000.0

# Backlog parameters
USE_BACKLOG = True
BACKLOG_AGE_PENALTY = 1.2

# Reputation Impact Parameters
REPUTATION_IMPACT = {
    'delay': {
        'base_cost': 1.0,        # Additional reputational cost per delayed order
        'severity_multiplier': 1.2
    },
    'defect': {
        'base_cost': 3.0,        # Additional reputational cost per defective item
        'volume_multiplier': 1.3
    },
    'spoilage': {
        'base_cost': 2.0,        # Additional reputational cost per spoiled item
        'volume_multiplier': 1.2
    }
}

# -----------------------------
# 2. DATA STRUCTURES
# -----------------------------

class CostBreakdown(NamedTuple):
    outage_costs: float      # Direct costs from warehouse outages
    delay_costs: float       # Direct compensation for delays
    defect_costs: float      # Direct costs for defects
    spoilage_costs: float    # Direct costs for spoilage
    reputation_costs: float  # Additional reputation impact costs
    
    @property
    def total(self) -> float:
        return (self.outage_costs + self.delay_costs + self.defect_costs +
                self.spoilage_costs + self.reputation_costs)

@dataclass
class Order:
    created_day: int
    age: int = 0
    
    @property
    def priority(self) -> float:
        """Higher priority for older orders."""
        return self.age * BACKLOG_AGE_PENALTY

@dataclass
class WarehouseState:
    time_to_outage: float
    outage_remaining: float = 0.0

# -----------------------------
# 3. UTILITY FUNCTIONS
# -----------------------------

def draw_outage_duration():
    """Draw an outage duration (in hours) from an exponential distribution."""
    return np.random.exponential(OUTAGE_DURATION_MEAN)

def draw_time_to_failure():
    """Draw time to next outage (in days) from an exponential distribution."""
    return np.random.exponential(MTBO_MEAN)

def sample_shipping_times(num_orders):
    """Generate shipping times from a lognormal distribution."""
    if num_orders <= 0:
        return np.array([])
    return np.random.lognormal(SHIPPING_TIME_MU, SHIPPING_TIME_SIGMA, num_orders)

def calculate_delay_metrics(shipping_times):
    """Calculate delay metrics for shipping times exceeding threshold."""
    delays = shipping_times[shipping_times > DELAY_THRESHOLD]
    if len(delays) > 0:
        delay_days = delays - DELAY_THRESHOLD  # Days beyond threshold
        return len(delays), np.sum(delay_days), delay_days
    return 0, 0, np.array([])

def calculate_delay_cost(delay_days: float, threshold: float) -> float:
    """
    Calculate direct compensation cost for a delayed order.
    
    Args:
        delay_days: Additional days beyond the threshold
        threshold: Delay threshold (e.g., 5 days)
    """
    total_days = delay_days + threshold
    for thresh, cost in sorted(DELAY_COSTS.items(), reverse=True):
        if total_days >= thresh:
            return cost
    return 0.0

def sample_defects(num_shipped):
    """Calculate the number of defects, including the possibility of clusters."""
    if num_shipped <= 0:
        return 0
    if np.random.random() < DEFECT_CLUSTER_PROBABILITY:
        defect_rate = BASE_DEFECT_RATE * DEFECT_CLUSTER_MULTIPLIER
    else:
        defect_rate = BASE_DEFECT_RATE
    return np.random.binomial(num_shipped, defect_rate)

def calculate_spoilage(shipping_times, current_day):
    """
    Calculate the number of spoiled items based on shipping times.
    Incorporates seasonal variation via a cosine-based factor.
    """
    seasonal_factor = 1.0 + SEASONAL_SPOILAGE_IMPACT * np.cos(
        2 * np.pi * (current_day % 365) / 365
    )
    adjusted_threshold = BASE_SPOILAGE_THRESHOLD * seasonal_factor
    return np.sum(shipping_times > adjusted_threshold)

def calculate_reputation_impact(delays: np.ndarray, num_defects: int, num_spoiled: int) -> float:
    """
    Calculate reputational impact based on severity and volume of issues.
    Uses dampened scaling to prevent extreme outliers.
    """
    reputation_cost = 0.0
    
    # Delays
    if len(delays) > 0:
        delay_severity = min(np.mean(delays), 10)  # Cap at 10 days
        delay_impact = (REPUTATION_IMPACT['delay']['base_cost'] *
                        len(delays) *
                        (1 + delay_severity * 0.1))
        reputation_cost += delay_impact
    
    # Defects
    if num_defects > 0:
        defect_scale = np.sqrt(num_defects / 100) if num_defects > 100 else 1
        defect_impact = (REPUTATION_IMPACT['defect']['base_cost'] *
                         num_defects *
                         min(1 + defect_scale * (REPUTATION_IMPACT['defect']['volume_multiplier'] - 1),
                             REPUTATION_IMPACT['defect']['volume_multiplier']))
        reputation_cost += defect_impact
    
    # Spoilage
    if num_spoiled > 0:
        spoilage_scale = np.sqrt(num_spoiled / 100) if num_spoiled > 100 else 1
        spoilage_impact = (REPUTATION_IMPACT['spoilage']['base_cost'] *
                           num_spoiled *
                           min(1 + spoilage_scale * (REPUTATION_IMPACT['spoilage']['volume_multiplier'] - 1),
                               REPUTATION_IMPACT['spoilage']['volume_multiplier']))
        reputation_cost += spoilage_impact
    
    return reputation_cost

# -----------------------------
# 4. SIMULATION FUNCTION
# -----------------------------

def run_simulation_scenario(n_iterations=N_ITERATIONS, days=SIMULATION_DAYS) -> Tuple[np.ndarray, List[Dict], List[CostBreakdown]]:
    """
    Runs a Monte Carlo simulation with enhanced risk and reputation modeling.
    Returns total costs, a list of metrics dicts, and a list of cost breakdowns.
    """
    total_costs = np.zeros(n_iterations)
    all_metrics = []
    all_cost_breakdowns = []

    for iteration in range(n_iterations):
        # Print progress occasionally
        if iteration % 30 == 0 and iteration > 0:
            print(f"Progress: Completed {iteration}/{n_iterations} iterations...")

        # Initialize warehouse states
        warehouse_state = [WarehouseState(draw_time_to_failure()) for _ in range(NUM_WAREHOUSES)]

        backlog = []
        iteration_metrics = {
            'max_backlog': 0,
            'avg_order_age': 0,
            'num_delayed_orders': 0,
            'total_delay_days': 0,
            'delay_days_per_order': [],
            'total_defects': 0,
            'total_spoiled': 0,
            'seasonal_spoilage': np.zeros(12)
        }

        # Track costs separately
        outage_costs = 0.0
        delay_costs = 0.0
        defect_costs = 0.0
        spoilage_costs = 0.0
        reputation_costs = 0.0

        for d in range(days):
            current_day = d

            # Generate daily order volume
            daily_orders = int(np.random.normal(BASE_DAILY_ORDERS, DAILY_ORDER_NORMAL_DIST_STD))
            daily_orders = max(0, daily_orders)

            # Age existing backlog
            for order in backlog:
                order.age += 1

            # Sort backlog by priority
            if backlog:
                backlog.sort(key=lambda x: x.priority, reverse=True)
                iteration_metrics['avg_order_age'] += sum(o.age for o in backlog) / len(backlog)

            # Determine daily warehouse capacity
            daily_warehouse_capacity = np.zeros(NUM_WAREHOUSES)
            for i, wh in enumerate(warehouse_state):
                fraction_outage = 0.0
                if wh.outage_remaining > 0:
                    if wh.outage_remaining >= OPERATING_HOURS_PER_DAY:
                        fraction_outage = 1.0
                        wh.outage_remaining -= OPERATING_HOURS_PER_DAY
                    else:
                        fraction_outage = wh.outage_remaining / OPERATING_HOURS_PER_DAY
                        wh.outage_remaining = 0.0
                elif d >= wh.time_to_outage:
                    duration_hrs = draw_outage_duration()
                    outage_costs += duration_hrs * OUTAGE_COST_PER_HOUR * WAREHOUSE_THROUGHPUT_SHARES[i]
                    if duration_hrs >= OPERATING_HOURS_PER_DAY:
                        fraction_outage = 1.0
                        wh.outage_remaining = duration_hrs - OPERATING_HOURS_PER_DAY
                    else:
                        fraction_outage = duration_hrs / OPERATING_HOURS_PER_DAY
                    wh.time_to_outage = d + draw_time_to_failure()

                fraction_operating = 1.0 - fraction_outage
                daily_warehouse_capacity[i] = WAREHOUSE_THROUGHPUT_SHARES[i] * daily_orders * fraction_operating

            daily_capacity = int(np.floor(np.sum(daily_warehouse_capacity)))

            # Add new orders to backlog
            new_orders = [Order(created_day=d) for _ in range(daily_orders)]
            backlog.extend(new_orders)

            iteration_metrics['max_backlog'] = max(iteration_metrics['max_backlog'], len(backlog))

            # Process (ship) orders up to capacity
            shipped_orders = backlog[:daily_capacity]
            backlog = backlog[daily_capacity:]
            shipped_count = len(shipped_orders)

            # Shipping times & delays
            shipping_times = sample_shipping_times(shipped_count)
            num_delayed, total_delay_days, delay_durations = calculate_delay_metrics(shipping_times)
            iteration_metrics['num_delayed_orders'] += num_delayed
            iteration_metrics['total_delay_days'] += total_delay_days
            if len(delay_durations) > 0 and d == days - 1:
                iteration_metrics['delay_days_per_order'] = delay_durations.tolist()

            # Delay costs
            for delay in delay_durations:
                delay_costs += calculate_delay_cost(delay, DELAY_THRESHOLD)

            # Defects
            defects_today = sample_defects(shipped_count)
            defect_costs += defects_today * COST_PER_DEFECT
            iteration_metrics['total_defects'] += defects_today

            # Spoilage
            spoiled_today = calculate_spoilage(shipping_times, current_day)
            spoilage_costs += spoiled_today * COST_PER_SPOILED
            iteration_metrics['total_spoiled'] += spoiled_today

            # Track spoilage by approximate month
            current_month = min((current_day % 365) // 30, 11)
            iteration_metrics['seasonal_spoilage'][current_month] += spoiled_today

            # Reputation
            reputation_costs += calculate_reputation_impact(delay_durations, defects_today, spoiled_today)

        # Summarize costs
        cost_breakdown = CostBreakdown(
            outage_costs=outage_costs,
            delay_costs=delay_costs,
            defect_costs=defect_costs,
            spoilage_costs=spoilage_costs,
            reputation_costs=reputation_costs
        )
        total_costs[iteration] = cost_breakdown.total

        if backlog:
            iteration_metrics['avg_order_age'] /= days

        all_metrics.append(iteration_metrics)
        all_cost_breakdowns.append(cost_breakdown)

    return total_costs, all_metrics, all_cost_breakdowns

# -----------------------------
# 5. STACKED PROPORTION PLOT
# -----------------------------
def plot_stacked_cost_proportions(total_costs, cost_df, bins=20):
    """
    Creates a 100%-stacked bar chart showing, for each bin of total cost,
    the proportion of each cost category.
    
    Args:
        total_costs: 1D array or Series of total cost (one entry per simulation).
        cost_df:     DataFrame with columns like 'outage_costs', 'delay_costs',
                     'defect_costs', 'spoilage_costs', 'reputation_costs'.
                     Must be the same length as total_costs.
        bins:        Number of bins to use along the total-cost axis.
    """
    # 1) Define bin edges
    bin_edges = np.linspace(total_costs.min(), total_costs.max(), bins+1)
    
    # 2) Digitize: find which bin each simulation belongs to
    bin_index = np.digitize(total_costs, bin_edges) - 1
    
    # 3) Accumulate sums of each cost category per bin
    cost_categories = ['outage_costs', 'delay_costs', 'defect_costs',
                       'spoilage_costs', 'reputation_costs']
    bin_sums = np.zeros((bins, len(cost_categories)))
    
    for i in range(len(total_costs)):
        b = bin_index[i]
        if b < 0 or b >= bins:
            # If a value == max(total_costs), it can fall outside the last bin
            continue
        for j, cat in enumerate(cost_categories):
            bin_sums[b, j] += cost_df[cat].iloc[i]
    
    # 4) Convert sums to proportions
    bin_totals = bin_sums.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_proportions = np.where(bin_totals > 0, bin_sums / bin_totals, 0)
    
    # 5) Plot stacked bars
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = bin_edges[1:] - bin_edges[:-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(bins)
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    labels = ['Outage', 'Delay', 'Defect', 'Spoilage', 'Reputation']
    
    for j, cat in enumerate(cost_categories):
        ax.bar(
            x_centers,
            bin_proportions[:, j],
            bottom=bottom,
            width=width * 0.9,  # slightly smaller than full bin width
            color=colors[j],
            label=labels[j]
        )
        bottom += bin_proportions[:, j]
    
    ax.set_xlabel("Total Cost (binned)")
    ax.set_ylabel("Proportion of Cost by Category")
    ax.set_title("Stacked Proportion of Cost Categories by Total Cost Bin")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. ANALYSIS AND PLOTTING
# -----------------------------

def analyze_results(costs: np.ndarray, metrics: List[Dict], cost_breakdowns: List[CostBreakdown], save_plot: bool = True):
    """
    Analyze and display simulation results with detailed cost breakdowns.
    """
    simulation_period = "year" if SIMULATION_DAYS == 365 else f"{SIMULATION_DAYS} days"
    
    # Convert cost breakdowns to DataFrame
    cost_df = pd.DataFrame(cost_breakdowns)
    metrics_df = pd.DataFrame(metrics)
    total_orders = BASE_DAILY_ORDERS * SIMULATION_DAYS
    
    print(f"\nRisk-Related Cost Analysis (per {simulation_period}):")
    print("\nBreakdown of Costs:")
    
    total_costs = cost_df.sum(axis=1)
    total_mean_cost = total_costs.mean()
    
    for cost_type in ['outage_costs', 'delay_costs', 'defect_costs', 'spoilage_costs', 'reputation_costs']:
        these_costs = cost_df[cost_type]
        print(f"\n{cost_type.replace('_', ' ').title()}:")
        print(f"  Mean: ${these_costs.mean():,.2f}")
        print(f"  Median: ${these_costs.median():,.2f}")
        print(f"  95th Percentile: ${these_costs.quantile(0.95):,.2f}")
        pct_of_total = (these_costs.mean() / total_mean_cost) * 100 if total_mean_cost > 0 else 0
        print(f"  % of Total Costs: {pct_of_total:.1f}%")
    
    print(f"\nTotal Risk-Related Costs:")
    print(f"Mean: ${total_costs.mean():,.2f}")
    print(f"Median: ${total_costs.median():,.2f}")
    print(f"95th Percentile: ${total_costs.quantile(0.95):,.2f}")
    print(f"99th Percentile: ${total_costs.quantile(0.99):,.2f}")

    # Operational metrics
    print(f"\nOperational Metrics (averaged across {N_ITERATIONS} iterations):")
    print("\nBacklog Metrics:")
    print(f"Average Maximum Backlog: {metrics_df['max_backlog'].mean():.1f} orders")
    print(f"Average Order Age: {metrics_df['avg_order_age'].mean():.1f} days")
    
    # Delay analysis
    avg_num_delayed = metrics_df['num_delayed_orders'].mean()
    print(f"\nDelay Analysis (>{DELAY_THRESHOLD} days):")
    print("Delayed Orders:")
    print(f"  Total: {avg_num_delayed:.1f} ({(avg_num_delayed/total_orders)*100:.1f}% of all orders)" if total_orders > 0 else "  None")
    print(f"  50th percentile: {metrics_df['num_delayed_orders'].quantile(0.5):.1f}")
    print(f"  95th percentile: {metrics_df['num_delayed_orders'].quantile(0.95):.1f}")
    
    avg_delay_days = (metrics_df['total_delay_days'].mean() / avg_num_delayed if avg_num_delayed > 0 else 0)
    print("\nDelay Duration:")
    print(f"  Average additional days beyond {DELAY_THRESHOLD}-day threshold: {avg_delay_days:.1f}")
    print(f"  Total delay days: {metrics_df['total_delay_days'].mean():.1f}")
    
    # Quality issues
    print("\nQuality Issues:")
    print(f"Defective Items: {metrics_df['total_defects'].mean():.1f}")
    print(f"Spoiled Items: {metrics_df['total_spoiled'].mean():.1f}")

    try:
        # If saving plots, switch to 'Agg' backend
        if save_plot:
            import matplotlib
            matplotlib.use('Agg')
        
        # 1. Create a figure for the total costs histogram, boxplot, etc.
        fig = plt.figure(figsize=(15, 15))
        
        # Subplot 1: Total costs histogram
        ax1 = plt.subplot(3, 1, 1)
        if total_costs.mean() > 1_000_000:
            plot_costs = total_costs / 1_000_000
            cost_unit = "Millions"
            cost_symbol = "M"
        else:
            plot_costs = total_costs / 1_000
            cost_unit = "Thousands"
            cost_symbol = "K"
        
        bins = 50
        ax1.hist(plot_costs, bins=bins, color='lightblue', edgecolor='black')
        ax1.set_title(f"Distribution of Operational Risk Costs Over {N_ITERATIONS} Simulations\n({simulation_period})")
        ax1.set_xlabel(f"Total Cost ({cost_unit} $)")
        ax1.set_ylabel("Number of Simulations")
        
        # Add percentile lines
        if len(plot_costs) > 0:
            median_cost = np.median(plot_costs)
            p95_cost = np.percentile(plot_costs, 95)
            ax1.axvline(median_cost, color='black', linestyle='--', alpha=0.5,
                        label=f'Median: ${median_cost:.2f}{cost_symbol}')
            ax1.axvline(p95_cost, color='red', linestyle='--', alpha=0.5,
                        label=f'95th %ile: ${p95_cost:.2f}{cost_symbol}')
            ax1.legend(loc='upper right')
        
        ax1.xaxis.set_major_formatter(lambda x, p: f'${x:.2f}{cost_symbol}')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Cost breakdown boxplot
        ax2 = plt.subplot(3, 1, 2)
        cost_columns = ['outage_costs', 'delay_costs', 'defect_costs', 'spoilage_costs', 'reputation_costs']
        if total_costs.mean() > 1_000_000:
            box_data = [cost_df[col] / 1_000_000 for col in cost_columns]
            cost_label = "Cost (Millions $)"
        else:
            box_data = [cost_df[col] / 1_000 for col in cost_columns]
            cost_label = "Cost (Thousands $)"
            
        tick_labels = [col.replace('_costs', '').title() for col in cost_columns]
        ax2.boxplot(box_data, labels=tick_labels)
        ax2.set_title(f"Annual Cost Breakdown by Risk Category)")
        ax2.set_ylabel(cost_label)
        ax2.yaxis.set_major_formatter(lambda x, p: f'${x:.2f}{cost_symbol}')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Delay distribution (from last day of each iteration)
        ax3 = plt.subplot(3, 1, 3)
        all_delays = []
        for m in metrics:
            all_delays.extend(m['delay_days_per_order'])
        
        if all_delays:
            ax3.hist(all_delays, bins=50, color='lightcoral', edgecolor='black')
            ax3.set_title(f"Distribution of Additional Days Beyond {DELAY_THRESHOLD}-day Threshold\n({simulation_period})")
            ax3.set_xlabel(f"Additional Days Beyond {DELAY_THRESHOLD}-day Threshold")
            ax3.set_ylabel("Number of Orders")
            if len(all_delays) > 0:
                median_delay = np.median(all_delays)
                p95_delay = np.percentile(all_delays, 95)
                ax3.axvline(median_delay, color='black', linestyle='--', alpha=0.5)
                ax3.axvline(p95_delay, color='red', linestyle='--', alpha=0.5)
                ax3.text(median_delay, ax3.get_ylim()[1], f'Median: {median_delay:.1f}d',
                         rotation=90, va='top', ha='right')
                ax3.text(p95_delay, ax3.get_ylim()[1], f'95th %ile: {p95_delay:.1f}d',
                         rotation=90, va='top', ha='right', color='red')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if save_plot:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'simulation_results_{timestamp}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"\nPlot saved as '{filename}'")
            plt.close()
        else:
            plt.show()
        
        # -- NEW: Plot stacked proportions --
        # Rerun or create a new figure for the stacked bar chart
        if save_plot:
            import matplotlib
            matplotlib.use('Agg')
        
        # Plot the stacked cost proportions
        plot_stacked_cost_proportions(total_costs, cost_df, bins=20)
        
        if save_plot:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_stacked = f'simulation_results_stacked_{timestamp}.png'
            plt.savefig(filename_stacked, bbox_inches='tight', dpi=300)
            print(f"\nStacked proportion plot saved as '{filename_stacked}'")
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {str(e)}")
        print("Results are still valid - only the plotting failed")


# -----------------------------
# 7. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    days = SIMULATION_DAYS
    iterations = N_ITERATIONS
    
    print(f"Running simulation for {days} days with {iterations} iterations...")
    costs, metrics, cost_breakdowns = run_simulation_scenario(n_iterations=iterations, days=days)
    analyze_results(costs, metrics, cost_breakdowns, save_plot=True)
