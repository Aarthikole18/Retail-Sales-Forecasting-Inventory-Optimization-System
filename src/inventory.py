import numpy as np

def calculate_inventory(predictions, actual_sales, lead_time=3, service_level=1.65):
    """
    predictions: predicted demand
    actual_sales: actual demand (for std deviation)
    lead_time: days to restock
    service_level: z-score (1.65 ≈ 95%)
    """

    # Demand variability
    demand_std = np.std(actual_sales)

    # Safety Stock formula
    safety_stock = service_level * demand_std * np.sqrt(lead_time)

    # Demand during lead time
    demand_lead_time = sum(predictions[:lead_time])

    # Reorder Point
    reorder_point = demand_lead_time + safety_stock

    return safety_stock, reorder_point