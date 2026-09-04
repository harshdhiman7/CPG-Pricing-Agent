import json
import pandas as pd
import ollama

# Choose your lightweight Qwen model (e.g., 'qwen2.5:3b' or 'qwen2.5:7b')
MODEL_NAME = "qwen2.5:3b"

# ==========================================
# 1. PRICING TOOLS (Deterministic Analytics)
# ==========================================

def calculate_price_elasticity(df: pd.DataFrame, upc: int) -> dict:
    """Calculates price elasticity of demand (PED) for a given product UPC."""
    product_data = df[df['UPC'] == upc].sort_values('WEEK_END_DATE')
    if product_data.empty or len(product_data) < 2:
        return {"error": "Insufficient data for given UPC"}

    product_data['pct_change_units'] = product_data['UNITS'].pct_change()
    product_data['pct_change_price'] = product_data['PRICE'].pct_change()

    valid_changes = product_data[(product_data['pct_change_price'] != 0) & (product_data['pct_change_price'].notna())]
    if valid_changes.empty:
        return {"elasticity": "Inelastic / Fixed Price Data", "avg_price": float(product_data['PRICE'].mean())}

    avg_elasticity = (valid_changes['pct_change_units'] / valid_changes['pct_change_price']).mean()
    
    return {
        "upc": int(upc),
        "product_name": str(product_data['DESCRIPTION'].iloc[0]),
        "average_elasticity": round(float(avg_elasticity), 2),
        "demand_sensitivity": "Elastic" if abs(avg_elasticity) > 1 else "Inelastic",
        "avg_units_sold": float(product_data['UNITS'].mean()),
        "current_price": float(product_data['PRICE'].iloc[-1])
    }

def analyze_promo_impact(df: pd.DataFrame, upc: int) -> dict:
    """Compares baseline vs feature/display promotion performance."""
    product_data = df[df['UPC'] == upc]
    if product_data.empty:
        return {"error": "UPC not found"}

    promo_mask = (product_data['FEATURE'] == 1) | (product_data['DISPLAY'] == 1) | (product_data['TPR_ONLY'] == 1)
    
    promo_units = product_data[promo_mask]['UNITS'].mean() if not product_data[promo_mask].empty else 0
    non_promo_units = product_data[~promo_mask]['UNITS'].mean() if not product_data[~promo_mask].empty else 0

    uplift_pct = (((promo_units - non_promo_units) / non_promo_units) * 100) if non_promo_units > 0 else 0

    return {
        "upc": int(upc),
        "non_promo_avg_units": round(float(non_promo_units), 2),
        "promo_avg_units": round(float(promo_units), 2),
        "promotional_uplift_pct": f"{round(float(uplift_pct), 2)}%"
    }

def get_store_segment_metrics(df: pd.DataFrame, store_id: int) -> dict:
    """Fetches high-level metrics for a specific retail store location."""
    store_data = df[df['STORE_ID'] == store_id]
    if store_data.empty:
        return {"error": "Store ID not found"}

    first_row = store_data.iloc[0]
    return {
        "store_id": int(store_id),
        "store_name": str(first_row['STORE_NAME']),
        "city": str(first_row['ADDRESS_CITY_NAME']),
        "state": str(first_row['ADDRESS_STATE_PROV_CODE']),
        "segment_type": str(first_row['SEG_VALUE_NAME']),
        "avg_weekly_baskets": float(first_row['AVG_WEEKLY_BASKETS']),
        "total_store_spend": float(store_data['SPEND'].sum())
    }

# Tool Registry
TOOL_MAP = {
    "calculate_price_elasticity": calculate_price_elasticity,
    "analyze_promo_impact": analyze_promo_impact,
    "get_store_segment_metrics": get_store_segment_metrics
}


# ==========================================
# 2. PRICING AGENT CLASS (Qwen-Optimized)
# ==========================================

class PricingAgent:
    def __init__(self, data_df: pd.DataFrame, model_name: str = MODEL_NAME):
        self.df = data_df
        self.model_name = model_name
        self.system_prompt = (
            "You are an expert Retail Pricing Strategy Agent powered by Qwen. "
            "Use the provided function tools to retrieve elasticity, promotion, and store data. "
            "Always call the relevant functions before synthesizing your final answer."
        )

    def run(self, user_query: str) -> str:
        """Processes query with Qwen, executes local tools, and returns the response."""
        
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'calculate_price_elasticity',
                    'description': 'Calculate the price elasticity of demand for a specific product UPC.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'upc': {'type': 'integer', 'description': 'Product UPC identifier'}
                        },
                        'required': ['upc']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_promo_impact',
                    'description': 'Analyze impact of displays, features, and temporary price reductions on volume.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'upc': {'type': 'integer', 'description': 'Product UPC identifier'}
                        },
                        'required': ['upc']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_store_segment_metrics',
                    'description': 'Get retail metadata and basket performance metrics for a store.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'store_id': {'type': 'integer', 'description': 'Unique store ID'}
                        },
                        'required': ['store_id']
                    }
                }
            }
        ]

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_query}
        ]

        # Call Qwen via Ollama API
        response = ollama.chat(model=self.model_name, messages=messages, tools=tools)

        # Process function execution if Qwen flags a tool call
        if response.get('message', {}).get('tool_calls'):
            messages.append(response['message'])

            for tool in response['message']['tool_calls']:
                fn_name = tool['function']['name']
                fn_args = tool['function']['arguments']

                if fn_name in TOOL_MAP:
                    # Execute Pandas tool locally
                    tool_result = TOOL_MAP[fn_name](self.df, **fn_args)

                    # Send result back to Qwen context
                    messages.append({
                        'role': 'tool',
                        'content': json.dumps(tool_result),
                    })

            # Get final answer from Qwen
            final_response = ollama.chat(model=self.model_name, messages=messages)
            return final_response['message']['content']
        
        return response['message']['content']


# ==========================================
# 3. EXECUTION DEMO
# ==========================================

if __name__ == "__main__":
    # Mock dataframe
    data = {
        'WEEK_END_DATE': ['2026-01-07', '2026-01-14', '2026-01-21', '2026-01-28'],
        'STORE_ID': [101, 101, 101, 101],
        'UPC': [123456, 123456, 123456, 123456],
        'UNITS': [120, 180, 110, 250],
        'VISITS': [500, 520, 480, 600],
        'HHS': [300, 310, 290, 350],
        'SPEND': [360.0, 450.0, 352.0, 500.0],
        'PRICE': [3.00, 2.50, 3.20, 2.00],
        'BASE_PRICE': [3.20, 3.20, 3.20, 3.20],
        'FEATURE': [0, 1, 0, 1],
        'DISPLAY': [0, 0, 0, 1],
        'TPR_ONLY': [0, 1, 0, 0],
        'DESCRIPTION': ['Organic Milk 1Gal', 'Organic Milk 1Gal', 'Organic Milk 1Gal', 'Organic Milk 1Gal'],
        'MANUFACTURER': ['DairyCo', 'DairyCo', 'DairyCo', 'DairyCo'],
        'CATEGORY': ['Dairy', 'Dairy', 'Dairy', 'Dairy'],
        'SUB_CATEGORY': ['Milk', 'Milk', 'Milk', 'Milk'],
        'PRODUCT_SIZE': ['1 GAL', '1 GAL', '1 GAL', '1 GAL'],
        'STORE_NAME': ['Main St Store', 'Main St Store', 'Main St Store', 'Main St Store'],
        'ADDRESS_CITY_NAME': ['Austin', 'Austin', 'Austin', 'Austin'],
        'ADDRESS_STATE_PROV_CODE': ['TX', 'TX', 'TX', 'TX'],
        'MSA_CODE': [12420, 12420, 12420, 12420],
        'SEG_VALUE_NAME': ['Upscale', 'Upscale', 'Upscale', 'Upscale'],
        'PARKING_SPACE_QTY': [150, 150, 150, 150],
        'SALES_AREA_SIZE_NUM': [25000, 25000, 25000, 25000],
        'AVG_WEEKLY_BASKETS': [4500, 4500, 4500, 4500]
    }

    df = pd.DataFrame(data)

    agent = PricingAgent(data_df=df, model_name=MODEL_NAME)

    query = "Analyze UPC 123456. What is its elasticity, promo impact, and recommended price strategy?"
    
    print(f"Running query with {MODEL_NAME}...\n")
    output = agent.run(query)
    print(output)