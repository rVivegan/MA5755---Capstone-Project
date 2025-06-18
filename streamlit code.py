import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import re

st.set_page_config(page_title="Coffee Shop Forecast", layout="wide")
st.title("☕ Cafe Coffee Day Forecasting and Inventory Optimization")

@st.cache_data
def load_data():
    sales_df = pd.read_csv("ccd_data.csv")
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])

    ingredients_df = pd.read_csv("ingredients.csv")

    return sales_df, ingredients_df

data, ingredients_df = load_data()

st.sidebar.header("⚙️ Forecast Settings")
all_items = sorted(data['Item Name'].unique().tolist())
forecast_options = ['All', 'Top 10 Items'] + all_items
selected_item = st.sidebar.selectbox("Select Item", forecast_options)
forecast_days = st.sidebar.slider("Days to Forecast", min_value=7, max_value=365, value=17)

model_choice = st.sidebar.selectbox("Select Prediction Model", ['Random Forest', 'Support Vector Regressor'])

last_date = data['Date'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
X_future = pd.DataFrame({
    'dayofweek': future_dates.dayofweek,
    'day': future_dates.day,
    'month': future_dates.month,
    'year': future_dates.year
})

def estimate_ingredients(item_name, predicted_quantity):
    ing_row = ingredients_df[ingredients_df['Item Name'] == item_name]
    if ing_row.empty:
        return pd.DataFrame(columns=["Ingredient", "Unit", "Total Amount"])
    
    try:
        raw_amounts = ing_row.iloc[0]['Ingredient Amounts (per item)']
        if pd.isna(raw_amounts) or not str(raw_amounts).strip():
            return pd.DataFrame(columns=["Ingredient", "Unit", "Total Amount"])
        
        ingredient_pairs = re.findall(r'([A-Za-z\s]+):\s*([\d.]+)\s*([a-zA-Z]+)', raw_amounts)
        if not ingredient_pairs:
            return pd.DataFrame(columns=["Ingredient", "Unit", "Total Amount"])

        ingredients = [name.strip() for name, _, _ in ingredient_pairs]
        amounts = [float(val) for _, val, _ in ingredient_pairs]
        units = [unit.strip() for _, _, unit in ingredient_pairs]
        total_required = [round(a * predicted_quantity, 2) for a in amounts]

        return pd.DataFrame({
            "Ingredient": ingredients,
            "Unit": units,
            "Total Amount": total_required
        })
    except Exception as e:
        st.warning(f"Error processing ingredients for item '{item_name}': {e}")
        return pd.DataFrame(columns=["Ingredient", "Unit", "Total Amount"])

if selected_item not in ['All', 'Top 10 Items']:
    item_data = data[data['Item Name'] == selected_item].copy()
    item_data = item_data.groupby('Date').agg({
        'Quantity': 'sum',
        'Rate': 'mean',
        'Discount_per_unit': 'mean'
    }).reset_index()

    item_data['dayofweek'] = item_data['Date'].dt.dayofweek
    item_data['day'] = item_data['Date'].dt.day
    item_data['month'] = item_data['Date'].dt.month
    item_data['year'] = item_data['Date'].dt.year

    X = item_data[['dayofweek', 'day', 'month', 'year']]
    y_qty = item_data['Quantity']

    model_qty = RandomForestRegressor() if model_choice == 'Random Forest' else SVR()
    model_qty.fit(X, y_qty)
    pred_qty = model_qty.predict(X_future)

    avg_rate = item_data['Rate'].mean()
    avg_discount = item_data['Discount_per_unit'].mean()
    pred_net_sales = (pred_qty * avg_rate) - (avg_discount * pred_qty)

    st.subheader(f"📦 Predicted Quantity for `{selected_item}`")
    st.line_chart(pd.DataFrame({'Date': future_dates, 'Predicted Quantity': pred_qty}).set_index('Date'))

    st.subheader(f"💰 Predicted Net Sales for `{selected_item}`")
    st.line_chart(pd.DataFrame({'Date': future_dates, 'Predicted Net Sales': pred_net_sales}).set_index('Date'))

    st.subheader("🧂 Estimated Ingredients Required")
    total_quantity = pred_qty.sum()
    st.dataframe(estimate_ingredients(selected_item, total_quantity))

elif selected_item == 'All':
    st.markdown(f"### 🔍 Prediction Summary for the Next `{forecast_days}` Days")
    quantity_results = []
    sales_results = []
    all_ingredients = {}

    for item in data['Item Name'].unique():
        item_data = data[data['Item Name'] == item].copy()
        item_daily = item_data.groupby('Date')[['Quantity', 'Rate', 'Discount_per_unit']].mean().reset_index()

        item_daily['dayofweek'] = item_daily['Date'].dt.dayofweek
        item_daily['day'] = item_daily['Date'].dt.day
        item_daily['month'] = item_daily['Date'].dt.month
        item_daily['year'] = item_daily['Date'].dt.year

        X_item = item_daily[['dayofweek', 'day', 'month', 'year']]
        y_qty = item_daily['Quantity']

        model_qty = RandomForestRegressor() if model_choice == 'Random Forest' else SVR()
        model_qty.fit(X_item, y_qty)
        pred_qty = model_qty.predict(X_future)
        total_pred_qty = pred_qty.sum()

        avg_rate = item_data['Rate'].mean()
        avg_discount = item_data['Discount_per_unit'].mean()
        pred_net_sales = (total_pred_qty * avg_rate) - (total_pred_qty * avg_discount)

        quantity_results.append({'Item Name': item, 'Predicted Quantity': round(total_pred_qty, 2)})
        sales_results.append({'Item Name': item, 'Predicted Net Sales': round(pred_net_sales, 2)})

        ingredient_df = estimate_ingredients(item, total_pred_qty)
        for _, row in ingredient_df.iterrows():
            key = (row['Ingredient'], row['Unit'])
            all_ingredients[key] = all_ingredients.get(key, 0) + row['Total Amount']

    df_qty = pd.DataFrame(quantity_results).sort_values(by='Predicted Quantity', ascending=False)
    df_sales = pd.DataFrame(sales_results).sort_values(by='Predicted Net Sales', ascending=False)

    st.subheader("📦 Predicted Quantity per Item")
    st.dataframe(df_qty)

    st.subheader("💰 Predicted Net Sales per Item")
    st.dataframe(df_sales)

    st.subheader("🧂 Total Estimated Ingredients Required for All Items")
    ingredient_summary = pd.DataFrame(
        [(k[0], k[1], round(v, 2)) for k, v in all_ingredients.items()],
        columns=['Ingredient', 'Unit', 'Total Amount']
    )
    st.dataframe(ingredient_summary)

elif selected_item == 'Top 10 Items':
    st.markdown(f"### 🔝 Top 10 Items by Forecasted Quantity (Next `{forecast_days}` Days)")
    top_items_df = []
    ingredient_summary = {}

    for item in data['Item Name'].unique():
        item_data = data[data['Item Name'] == item].copy()
        item_daily = item_data.groupby('Date')[['Quantity', 'Rate', 'Discount_per_unit']].mean().reset_index()

        item_daily['dayofweek'] = item_daily['Date'].dt.dayofweek
        item_daily['day'] = item_daily['Date'].dt.day
        item_daily['month'] = item_daily['Date'].dt.month
        item_daily['year'] = item_daily['Date'].dt.year

        X_item = item_daily[['dayofweek', 'day', 'month', 'year']]
        y_qty = item_daily['Quantity']

        model_qty = RandomForestRegressor() if model_choice == 'Random Forest' else SVR()
        model_qty.fit(X_item, y_qty)
        pred_qty = model_qty.predict(X_future)
        total_pred_qty = pred_qty.sum()

        avg_rate = item_data['Rate'].mean()
        avg_discount = item_data['Discount_per_unit'].mean()
        pred_net_sales = (total_pred_qty * avg_rate) - (total_pred_qty * avg_discount)

        top_items_df.append({
            'Item Name': item,
            'Total Forecast Quantity': round(total_pred_qty, 2),
            'Total Forecast Net Sales': round(pred_net_sales, 2)
        })

    top10_df = pd.DataFrame(top_items_df).sort_values(by='Total Forecast Quantity', ascending=False).head(10)
    st.dataframe(top10_df.reset_index(drop=True))

    st.subheader("🧂 Estimated Ingredients Required for Top 10 Items")
    for item in top10_df['Item Name']:
        pred_qty = top10_df[top10_df['Item Name'] == item]['Total Forecast Quantity'].values[0]
        ing_df = estimate_ingredients(item, pred_qty)
        for _, row in ing_df.iterrows():
            key = (row['Ingredient'], row['Unit'])
            ingredient_summary[key] = ingredient_summary.get(key, 0) + row['Total Amount']

    total_ing_df = pd.DataFrame(
        [(k[0], k[1], round(v, 2)) for k, v in ingredient_summary.items()],
        columns=['Ingredient', 'Unit', 'Total Amount']
    )
    st.dataframe(total_ing_df)

