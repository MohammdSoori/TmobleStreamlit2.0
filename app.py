import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from datetime import datetime, timedelta, date
import jdatetime  # For Jalali to Gregorian conversion
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import requests
import json
import re
import unicodedata
from datetime import datetime, timedelta
import plotly.graph_objects as go
import statsmodels.api as sm  # for regression
import openpyxl

# Replace with your actual API key
# api_key = st.secrets["api_key"]
api_key = st.secrets["api_key"]
# Base URL for the API
base_url = 'https://app.didar.me/api'

# Function to convert Jalali dates to Gregorian
@st.cache_data
def jalali_to_gregorian_vectorized(date_series):
    def convert(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            year, month, day = map(int, date_str.split('/'))
            return jdatetime.date(year, month, day).togregorian()
        except:
            return pd.NaT
    return date_series.apply(convert)

@st.cache_data
def extract_vip_status(name_series):
    import unicodedata
    import pandas as pd

    # 1) Fill NaNs with empty string so we can operate safely
    name_series = name_series.fillna("")

    # 2) Normalize Unicode to canonical form (NFC)
    name_series = name_series.apply(lambda x: unicodedata.normalize('NFC', x))

    # 3) Replace Excel’s special code for 💎
    name_series = name_series.str.replace(r"_xD83D__xDC8E", "💎", regex=True)

    # 4) Remove potential zero-width or variation selectors (like U+200D, U+FE0F, etc.)
    name_series = name_series.str.replace(r"[\u200B-\u200D\uFE0F]", "", regex=True)

    # 5) Final VIP status check
    def get_vip_status(name):
        if not name or pd.isna(name):
            return 'Non-VIP'
        if '💎' in name:
            return 'Gold VIP'
        elif '⭐' in name:
            return 'Silver VIP'
        elif '💠' in name:
            return 'Bronze VIP'
        else:
            return 'Non-VIP'

    return name_series.apply(get_vip_status)

############# PRICE ELASTICITY PAGE #############




############################################################################
# OPTIONAL HELPER: If you already have a function that adds the Probability_of_Churn,
# CLTV, etc. to your RFM data, you can reuse it instead of defining it again.
############################################################################
@st.cache_data
def add_churn_metrics(df_in):
    """
    Example function to add Probability_of_Churn, CLTV, CRR, and a Recency_norm column
    to the existing RFM DataFrame. This is a placeholder; in practice, you'd replace it
    with a real model or real computations.
    """
    df = df_in.copy()
    if df.empty:
        return df

    # Ensure columns exist
    required_cols = ["Recency", "Frequency", "Monetary"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0

    # 1) Create a synthetic churn label: e.g., churn if Recency > 200
    df['churn_label'] = np.where(df['Recency'] > 200, 1, 0)

    # Train a quick logistic regression (toy example)
    from sklearn.linear_model import LogisticRegression
    X = df[['Recency', 'Frequency', 'Monetary']].fillna(0)
    y = df['churn_label']

    if len(X['Recency'].unique()) > 1:  # at least some variance
        model = LogisticRegression()
        model.fit(X, y)
        df['Probability_of_Churn'] = model.predict_proba(X)[:, 1]
    else:
        # Fallback if no variance
        df['Probability_of_Churn'] = 0.5

    # 2) Customer_Lifespan: naive approach = 1 / Probability_of_Churn
    df['Customer_Lifespan'] = np.where(
        df['Probability_of_Churn'] < 0.01,
        500,  # cap for near-zero churn
        1.0 / df['Probability_of_Churn']
    )

    # 3) CLTV: naive approach = Monetary * Frequency * Customer_Lifespan
    df['CLTV'] = df['Monetary'] * df['Frequency'] * df['Customer_Lifespan']

    # 4) CRR = 1 - Probability_of_Churn
    df['CRR'] = 1.0 - df['Probability_of_Churn']

    # 5) Recency_norm using MinMax scaling, then invert so “more recent” is higher
    scaler = MinMaxScaler()
    if df['Recency'].nunique() > 1:
        df['Recency_norm'] = scaler.fit_transform(df[['Recency']])
        df['Recency_norm'] = 1 - df['Recency_norm']  # invert
    else:
        df['Recency_norm'] = 1  # if no variance, everything is 1

    # Drop the synthetic label
    df.drop(columns=['churn_label'], inplace=True)

    return df

############################################################################
# MAIN PAGE FUNCTION: "Churned Analysis"
############################################################################
def churned_analysis_page(rfm_data_original: pd.DataFrame):
    """
    Streamlit page for analyzing and exporting customers in:
      - 'Churned'
      - 'Lost Big Spenders'
      - 'Big Loss'
      - 'At Risk'
    segments, with further clustering by Probability_of_Churn, CLTV, Recency norm, etc.
    """
    st.title("Churned Analysis")

    # 1) Filter to the relevant RFM segments
    target_segments = ["Churned", "Lost Big Spenders", "Big Loss", "At Risk"]
    churn_df = rfm_data_original[rfm_data_original['RFM_segment_label'].isin(target_segments)].copy()

    if churn_df.empty:
        st.warning("No customers found in the specified churn-related segments.")
        return

    # 2) Add Probability_of_Churn, CLTV, Recency_norm, etc. (if not already present)
    churn_df = add_churn_metrics(churn_df)

    # 3) Let the user filter by these additional metrics
    st.subheader("Filter by Probability of Churn and CLTV")

    col1, col2 = st.columns(2)
    with col1:
        min_churn = st.slider(
            "Minimum Probability of Churn",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
    with col2:
        min_cltv = st.number_input(
            "Minimum CLTV",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

    # 4) Apply the filters
    filtered_df = churn_df[
        (churn_df['Probability_of_Churn'] >= min_churn) &
        (churn_df['CLTV'] >= min_cltv)
    ]

    st.markdown(f"**Number of customers after filters:** {len(filtered_df)}")

    if filtered_df.empty:
        st.info("No customers match these filter criteria.")
        return

    # 5) Display Table
    st.dataframe(filtered_df[[
        'Customer ID', 'First Name', 'Last Name', 'Phone Number',
        'RFM_segment_label', 'Recency', 'Recency_norm',
        'Frequency', 'Monetary', 'Probability_of_Churn',
        'Customer_Lifespan', 'CLTV', 'CRR'
    ]].reset_index(drop=True))

    # 6) Simple scatter: Probability_of_Churn vs. CLTV
    st.subheader("Visualization: Probability of Churn vs. CLTV")
    fig = px.scatter(
        filtered_df,
        x='Probability_of_Churn',
        y='CLTV',
        color='RFM_segment_label',
        size='Monetary',
        hover_data=['Customer ID', 'First Name', 'Last Name'],
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Churn vs. CLTV (size = Monetary)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 7) Export buttons
    st.subheader("Export Filtered Data")

    def convert_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    def convert_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='ChurnedAnalysis')
        return output.getvalue()

    csv_data = convert_to_csv(filtered_df)
    excel_data = convert_to_excel(filtered_df)

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name="churned_analysis.csv",
            mime="text/csv"
        )
    with colB:
        st.download_button(
            label="Download as Excel",
            data=excel_data,
            file_name="churned_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.success("Churned Analysis completed.")



def price_elasticity_page(main_data):
    """
    A Streamlit page to analyze Price Elasticity by computing
    average Arc Elasticity from consecutive day pairs.
    
    It expects:
      1) `main_data` DataFrame: the main deals file already preprocessed.
         Must have columns:
             - 'تاریخ انجام معامله': the Gregorian date of the deal
             - 'قیمت': numeric price in that deal
             - 'عنوان محصول': product name in Persian
      2) The user to upload the second file containing daily data per product code,
         with columns:
             - 'Date': The date in Gregorian
             - columns named by product code, containing a measure (e.g., occupancy).
    """

    st.header("Price Elasticity Analysis (Arc Elasticity)")

    # ----------------------- 1) Define Product Name → Code Mapping -----------------------
    product_code_map = {
        "ونک ۲ خواب F استاندارد": "VanakF",
        "ولنجک A (بدون لباسشویی)": "VlnjkA",
        "مرزداران (استاندارد) C": "MrzC",
        "پارک وی ۷۰ متری A": "PrkwA",
        "شریعتی (پاسداران) تیپ ۲": "Shrt2",
        "بهشتی ۱ خواب جکوزی‌دار A": "BshtA",
        "ولیعصر A استاندارد": "VlA",
        "مرزداران (مستردار) A": "MrzA",
        "میرداماد دیزاین مدرن CF": "MrdCFModern",
        "میرداماد دیزاین صنعتی ۹۰ متری CF": "MrdICFndust90",
        "میرداماد دیزاین مینیمال CF": "MrdICFMinimal",
        "شریعتی (پاسداران) تیپ ۱": "Shrt1",
        "جمهوری ۲خواب B": "NflB",
        "میرداماد دیزاین نئوکلاسیک ۶۳ متری CF": "MrdICFNeoClassic63",
        "بهشتی ۲ خواب B": "BshtB",
        "ولیعصر B ویژه (بالکن‌دار)": "VlB",
        "کشاورز B استاندارد": "KshB",
        "پارک وی ۱۰۵ متری VIP": "PrkwVIP",
        "کشاورز(بدون لباسشویی) A": "KshA",
        "جمهوری ۱خواب A": "NflA",
        "ونک ۱ خواب C استاندارد": "VanakC",
        "ولنجک B (استاندارد)": "VlnjkB",
        "جردن ۸۵ متری B (اکونومی)": "JrdB",
        "پارک وی ۸۰ متری B": "PrkwB",
        "جمهوری ۲خواب D": "NflD",
        "بهشتی ۱ خواب (پذیرش پت) C": "BshtC",
        "جردن ۸۵ متری C جنوبی (ویژه)": "JrdC",
        "میرداماد دیزاین صنعتی ۷۵ متری CF": "MrdICFndust75",
        "میرداماد دیزاین نئوکلاسیک ۸۰ متری CF": "MrdICFNeoClassic80",
        "کوروش (استاندارد) A": "KorA",
        "جردن ۹۰ متری A (استاندارد)": "JrdA",
        "پارک وی C (پذیرش پت)": "PrkwC",
        "جمهوری ۲خواب E": "NflE",
        "میرداماد دیزاین نئوکلاسیک ۸۰ متری PF": "MrdIPFNeoClassic80",
        "جردن D (پذیرش پت)": "JrdD",
        "کوروش (ویژه) B": "KorB",
        "کشاورز C ویژه": "KshC",
        "میرداماد دیزاین صنعتی ۹۰ متری VIP2": "MrdIPFVIP2",
        "میرداماد دیزاین مینیمال PF": "MrdIPFMinimal",
        "ولنجک C (ویو شهر)": "VlnjkC",
        "میرداماد دیزاین مدرن PF": "MrdIPFModern",
        "میرداماد دیزاین صنعتی ۷۵ متری VIP1": "MrdIPFVIP1",
        "ونک ۱ خواب B ویژه": "VanakB",
        "بهشتی ۱ خواب جکوزی‌دار VIP": "BshtVIP1",
        "جمهوری ۲خواب C": "NflC",
        "بهشتی ۲ خواب - VIP2": "BshtVIP2",
        "میرداماد دیزاین نئوکلاسیک ۶۳ متری PF": "MrdIPFNeoClassic63",
        "ونک ۱ خواب A اکونومی": "VanakA",
        "ونک ۲ خواب D اکونومی": "VanakD",
        "ترنج ۲ خواب (مستر) C": "TrnjC",
        "ترنج ۲ خواب (مستر) ‌‌E": "TrnjE",
        "ترنج ۲ خواب B": "TrnjB",
        "ترنج ۲ خواب A": "TrnjA",
        "ترنج ۲ خواب (مستر) D": "TrnjD",
        "ونک ۲ خواب E ویژه": "VanakE",
    }

    # ----------------------- 2) File Uploader for the Second File -----------------------
    st.markdown("""
    **Step 1:** Upload your second file which must contain:
    - A column named **'Date'** (Gregorian date),
    - Other columns named after product codes (e.g. 'VlnjkA', 'VanakF', etc.) 
      containing some measure (e.g. occupancy).
    """)
    second_file = st.file_uploader("Choose the second file (XLSX or CSV)", type=["xlsx","csv"])

    # If no file, just stop here
    if not second_file:
        st.info("Please upload your second file to proceed.")
        return

    # Load second file
    try:
        file_ext = second_file.name.split('.')[-1].lower()
        if file_ext == "xlsx":
            df2 = pd.read_excel(second_file)
        else:
            df2 = pd.read_csv(second_file)

        # Ensure 'Date' is DateTime
        df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
    except Exception as e:
        st.error(f"Error reading the second file: {e}")
        return

    st.success("Second file loaded successfully!")
    st.write("Preview of the second file:")
    st.dataframe(df2.head())

    # ----------------------- 3) Prepare Main Data for Daily Avg Price by Code -----------------------
    # Make a copy of main_data so we don’t mutate the original
    temp_main = main_data.copy()

    # Ensure the deal date is in Gregorian datetime (should already be done in load_data, but just in case)
    temp_main['تاریخ انجام معامله'] = pd.to_datetime(temp_main['تاریخ انجام معامله'], errors='coerce')

    # Map Persian product names → product codes
    def map_to_code(name):
        name = str(name).strip()
        return product_code_map.get(name, None)  # or some fallback if not found

    temp_main['product_code'] = temp_main['عنوان محصول'].apply(map_to_code)

    # Filter out rows where product_code is None (not in dictionary)
    temp_main = temp_main[temp_main['product_code'].notna()]

    # For each date & product_code, compute average 'قیمت'
    temp_main.rename(columns={'تاریخ انجام معامله': 'DealDate'}, inplace=True)
    daily_price = (
        temp_main
        .groupby(['DealDate', 'product_code'], as_index=False)['قیمت']
        .mean()
        .rename(columns={'قیمت': 'AvgPrice'})
    )

    # ----------------------- 4) Merge with the Second File -----------------------
    # The second file presumably has columns: ['Date', 'VanakF', 'VlnjkA', ...]
    # We'll melt it into a long format for merging.

    # Exclude the 'Date' column from the melt
    melt_cols = [col for col in df2.columns if col != 'Date']

    # Melt the second file: each row = (Date, product_code, measure)
    long_df2 = df2.melt(
        id_vars='Date',
        value_vars=melt_cols,
        var_name='product_code',
        value_name='Measure'  # e.g. occupancy measure
    )

    # Merge daily_price with long_df2 on (Date, product_code)
    # daily_price has 'DealDate' as date, rename it to 'Date' for merging
    daily_price.rename(columns={'DealDate': 'Date'}, inplace=True)
    merged_df = pd.merge(daily_price, long_df2, on=['Date','product_code'], how='inner')

    st.subheader("Merged Data Preview")
    st.write("Here is how the daily average price merges with your second file’s measure:")
    st.dataframe(merged_df.head(15))

    # ----------------------- 5) Arc Elasticity Computation -----------------------
    st.markdown("""
    **Step 2:** Select the product code(s) below to analyze **Arc Elasticity** from consecutive days.
    """)

    all_codes = sorted(merged_df['product_code'].unique())
    selected_codes = st.multiselect("Select product codes:", all_codes, default=all_codes[:1])
    if not selected_codes:
        st.warning("No product codes selected.")
        return

    def compute_average_arc_elasticity(subdf, price_col='AvgPrice', measure_col='Measure'):
        """
        Computes the average Arc Elasticity from consecutive-day pairs
        for the given DataFrame subdf (already filtered to one product_code).

        Arc Elasticity formula (between two points p1,q1 and p2,q2):
          E_arc = ((q2 - q1) / ((q1 + q2)/2)) / ((p2 - p1) / ((p1 + p2)/2))

        Steps:
          1) Sort by Date.
          2) For each consecutive pair (day i, day i+1), compute arc elasticity.
          3) Return the average of all valid pairs. If no valid pair, return None.
        """
        # Drop rows with NaN or non-positive values
        valid = subdf.dropna(subset=[price_col, measure_col, 'Date']).copy()
        valid = valid[(valid[price_col] > 0) & (valid[measure_col] > 0)]

        # Sort by Date
        valid.sort_values('Date', inplace=True)

        # We'll accumulate arc elasticity for consecutive days
        arc_values = []
        rows = valid.to_dict('records')

        for i in range(len(rows) - 1):
            p1, q1 = rows[i][price_col], rows[i][measure_col]
            p2, q2 = rows[i+1][price_col], rows[i+1][measure_col]

            # Avoid dividing by zero if p1 + p2 = 0 or q1 + q2 = 0
            if (p1 + p2) <= 0 or (q1 + q2) <= 0:
                continue

            # If p1 == p2 or q1 == q2 exactly, the arc formula can still be computed,
            # but might yield 0 or a degenerate value. We'll just let it proceed.
            numerator = (q2 - q1) / ((q1 + q2) / 2.0)
            denominator = (p2 - p1) / ((p1 + p2) / 2.0)

            # If denominator is 0, skip
            if abs(denominator) < 1e-12:
                continue

            e_arc = numerator / denominator
            arc_values.append(e_arc)

        if len(arc_values) == 0:
            return None
        return np.mean(arc_values)

    # For each selected code, compute and display
    for code in selected_codes:
        st.write("---")
        st.subheader(f"Product Code: `{code}`")

        subdf = merged_df[merged_df['product_code'] == code].copy()
        elasticity = compute_average_arc_elasticity(subdf)

        if elasticity is None:
            st.warning("Not enough valid consecutive-day pairs to compute arc elasticity.")
            continue

        st.write(f"**Average Arc Elasticity**: `{elasticity:.4f}`")

        # Plot measure vs. price on a simple scatter
        fig = px.scatter(
            subdf,
            x='AvgPrice',
            y='Measure',
            title=f"Measure vs. AvgPrice for {code}",
            labels={'AvgPrice': 'Average Price', 'Measure': 'Measure'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Arc Elasticity Interpretation**:  
    \- Negative values: as price goes up, measure (e.g. occupancy) goes down.  
    \- Magnitude > 1: the measure is elastic (strong response).  
    \- Magnitude < 1: inelastic (weak response).  
    """)



@st.cache_data
def extract_blacklist_status(name_series):
    def get_blacklist_status(name):
        if pd.isna(name):
            return 'Non-BlackList'
        # بررسی وجود (*) در انتهای نام
        if re.search(r'\(\*\)\s*$', name):
            return 'BlackList'
        else:
            return 'Non-BlackList'
    return name_series.apply(get_blacklist_status)



#############################



# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    # Load the Excel file
    data = pd.read_excel(uploaded_file)

    # List of columns containing Jalali dates
    date_columns = [
        'تاریخ انجام معامله', 'تاریخ ایجاد معامله', 'تاریخ احتمالی انجام معامله',
        'تاریخ ورود', 'تاریخ خروج', 'شروع قرارداد', 'پایان قرارداد'
        # Add any additional date columns here
    ]

    # Convert Jalali dates to Gregorian
    for col in date_columns:
        if col in data.columns:
            data[col] = jalali_to_gregorian_vectorized(data[col])
            # Ensure the date columns are datetime objects
            data[col] = pd.to_datetime(data[col], errors='coerce')
        else:
            st.warning(f"Column '{col}' not found in the data.")

    # Clean 'تعداد شب' column by removing non-digit characters
    if 'تعداد شب' in data.columns:
        data['تعداد شب'] = data['تعداد شب'].astype(str).str.replace(r'[^\d.]', '', regex=True)  # Keep digits and decimal points
        data['تعداد شب'] = pd.to_numeric(data['تعداد شب'], errors='coerce')
        # Remove entries where 'تعداد شب' is unreasonably large (e.g., greater than 365)
        data.loc[data['تعداد شب'] > 700, 'تعداد شب'] = np.nan
    else:
        st.warning("Column 'تعداد شب' not found in the data.")

    # Similarly, convert 'ارزش معامله' to numeric
    if 'ارزش معامله' in data.columns:
        data['ارزش معامله'] = pd.to_numeric(data['ارزش معامله'], errors='coerce')
    else:
        st.warning("Column 'ارزش معامله' not found in the data.")

    # Extract VIP Status
    data['VIP Status'] = extract_vip_status(data['نام خانوادگی شخص معامله'])


#############################################

    # تبدیل ستون 'عنوان محصول' به رشته و پر کردن مقادیر نامعتبر
    data['عنوان محصول'] = data['عنوان محصول'].fillna('').astype(str)

    def extract_complex(row):
        if re.search(r'\bمیرداماد\b', row):
            return 'میرداماد'
        elif re.search(r'\bپارک وی\b', row):
            return 'پارک وی'
        elif re.search(r'\bولنجک\b', row):
            return 'ولنجک'
        elif re.search(r'\bبهشتی\b', row):
            return 'بهشتی'
        elif re.search(r'\bجردن\b', row):
            return 'جردن'
        elif re.search(r'\bمرزداران\b', row):
            return 'مرزداران'
        elif re.search(r'\bاقدسیه\b', row):
            return 'اقدسیه'
        elif re.search(r'\bجمهوری\b', row):
            return 'جمهوری'
        elif re.search(r'\bکشاورز\b', row):
            return 'کشاورز'
        elif re.search(r'\bترنج\b', row):
            return 'ترنج'
        elif re.search(r'\bویلا\b', row):
            return 'ویلا'
        elif re.search(r'\bونک\b', row):
            return 'ونک'
        elif re.search(r'\bکوروش\b', row):
            return 'کوروش'
        elif re.search(r'\bشریعتی\b', row):
            return 'شریعتی'
        elif re.search(r'\bولیعصر\b', row):
            return 'ولیعصر'
        elif re.search(r'\bوزرا\b', row):
            return 'وزرا'
        else:
            return 'نامشخص'

    # اضافه کردن ستون جدید برای مجتمع
    data['Complex'] = data['عنوان محصول'].apply(extract_complex)


    data['BlackList Status'] = extract_blacklist_status(data['نام خانوادگی شخص معامله'])




#########################################

    return data

@st.cache_data
def update_last_name(last_name, new_vip_status):
    # Define the mapping between VIP status and emoji
    vip_emoji_map = {
        'Gold VIP': '💎',
        'Silver VIP': '⭐',
        'Bronze VIP': '💠'
    }
    
    # Remove existing VIP-related emoji and text in parentheses
    last_name = re.sub(r'\s*\((💎|⭐|💠)?\s*VIP\s*\)', '', last_name).strip()
    
    # If new VIP status is Non-VIP, return the updated last name
    if new_vip_status == 'Non-VIP':
        return last_name
    
    # Add the new VIP emoji in parentheses at the end of the last name
    emoji = vip_emoji_map.get(new_vip_status, '')
    if emoji:
        last_name = f"{last_name} ({emoji}VIP)"
    
    return last_name

# Define the function to update the contact's last name via API
@st.cache_data
def update_contact_last_name(phone_number, updated_last_name):
    try:
        # Endpoint for searching contacts
        search_endpoint = '/contact/personsearch'
    
        # Full URL with API key for search
        search_url = f"{base_url}{search_endpoint}?apikey={api_key}"
    
        # Request payload for searching the contact
        search_payload = {
            "Criteria": {
                "IsDeleted": 0,
                "IsPinned": -1,
                "IsVIP": -1,
                "LeadType": -1,
                "Pin": -1,
                "SortOrder": 1,
                "Keywords": phone_number,
                "OwnerId": "00000000-0000-0000-0000-000000000000",
                "SearchFromTime": "1930-01-01T00:00:00.000Z",
                "SearchToTime": "9999-12-01T00:00:00.000Z",
                "CustomFields": [],
                "FilterId": None
            },
            "From": 0,
            "Limit": 30
        }
    
        # Headers
        headers = {
            'Content-Type': 'application/json'
        }
    
        # Step 1: Search for the contact
        response = requests.post(search_url, headers=headers, json=search_payload)
    
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON
            response_data = response.json()
            contacts = response_data.get('Response', {}).get('List', [])
            
            if contacts:
                # Assuming the first contact is the desired one
                contact = contacts[0]
                contact_id = contact.get('Id')
                
                # Update the contact's LastName
                contact['LastName'] = updated_last_name
                # Update DisplayName if necessary
                contact['DisplayName'] = (contact.get('FirstName', '') + ' ' + updated_last_name).strip()
                
                # Remove read-only or unnecessary fields
                fields_to_remove = [
                    'CanDelete', 'CanEdit', 'IsMine', 'HasAccess', '_Type', 'OwnerId_Old', 'Segments', 
                    'Owner', 'ContactStatus', 'KeepInTouch', 'Fields'
                ]
                for field in fields_to_remove:
                    contact.pop(field, None)
    
                # If 'Segments' are present, extract 'SegmentIds'
                segments = contacts[0].get('Segments', [])
                segment_ids = [segment.get('Id') for segment in segments]
    
                # Prepare the save payload
                save_payload = {
                    "Contact": contact,
                    "SegmentIds": segment_ids
                }
    
                # Endpoint to save/update the contact
                save_endpoint = '/contact/save'
                save_url = f"{base_url}{save_endpoint}?ApiKey={api_key}"
    
                # Make the POST request to save the updated contact
                save_response = requests.post(save_url, headers=headers, json=save_payload)
    
                if save_response.status_code == 200:
                    return True
                else:
                    print(f"Failed to update contact. Status code: {save_response.status_code}")
                    print(f"Response: {save_response.text}")
                    return False
            else:
                print("No contact found with the given Phone Number.")
                return False
        else:
            print(f"Failed to search for contact. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Function to calculate RFM

@st.cache_data
def calculate_rfm(data, today=None):
    # Divide 'ارزش معامله' by 10 as per the new requirement
    data['ارزش معامله'] = data['ارزش معامله'] / 10

    # Filter for successful deals
    successful_deals = data[data['وضعیت معامله'] == 'موفق']

    # Define today's date for recency calculation
    if today is None:
        today = datetime.today()
    else:
        today = pd.to_datetime(today)

    # Group by unique customer ID while including personal details
    rfm_data = successful_deals.groupby('کد دیدار شخص معامله').agg({
        'نام شخص معامله': 'first',
        'نام خانوادگی شخص معامله': 'first',
        'موبایل شخص معامله': 'first',
        'تاریخ انجام معامله': lambda x: (today - pd.to_datetime(x).max()).days,  # Recency
        'کد دیدار معامله': 'count',  # Frequency
        'ارزش معامله': 'sum',  # Monetary
        'تعداد شب': 'sum',  # Total Nights
        'VIP Status': 'first'  # VIP Status
    }).reset_index()

    # Rename columns for clarity
    rfm_data.rename(columns={
        'کد دیدار شخص معامله': 'Customer ID',
        'نام شخص معامله': 'First Name',
        'نام خانوادگی شخص معامله': 'Last Name',
        'موبایل شخص معامله': 'Phone Number',
        'تاریخ انجام معامله': 'Recency',
        'کد دیدار معامله': 'Frequency',
        'ارزش معامله': 'Monetary',
        'تعداد شب': 'Total Nights',
    }, inplace=True)

    # Compute average stay
    rfm_data['average stay'] = rfm_data['Total Nights'] / rfm_data['Frequency']

    # Compute Is Monthly
    rfm_data['Is Monthly'] = rfm_data['average stay'] > 15

    # Get last successful deal per customer
    last_deals = successful_deals.sort_values('تاریخ انجام معامله').groupby('کد دیدار شخص معامله').tail(1)

    # Merge 'تاریخ ورود' and 'تاریخ خروج' into 'rfm_data'
    rfm_data = rfm_data.merge(
        last_deals[['کد دیدار شخص معامله', 'تاریخ ورود', 'تاریخ خروج']],
        left_on='Customer ID',
        right_on='کد دیدار شخص معامله',
        how='left'
    )

    # Compute 'Is staying'
    rfm_data['Is staying'] = (today >= rfm_data['تاریخ ورود']) & (today <= rfm_data['تاریخ خروج'])

    # Drop the extra 'کد دیدار شخص معامله' column
    rfm_data.drop(columns=['کد دیدار شخص معامله'], inplace=True)

    # -------------------- New Code to Add Favorite Product and Last Product --------------------

    # Favorite Product: Product with the most successful deals per customer
    favorite_product = successful_deals[successful_deals['عنوان محصول'].notna()]
    favorite_product = favorite_product.groupby(['کد دیدار شخص معامله', 'عنوان محصول']).size().reset_index(name='DealCount')
    favorite_product = favorite_product.sort_values(['کد دیدار شخص معامله', 'DealCount'], ascending=[True, False])
    favorite_product = favorite_product.groupby('کد دیدار شخص معامله').first().reset_index()
    favorite_product = favorite_product[['کد دیدار شخص معامله', 'عنوان محصول']].rename(columns={'عنوان محصول': 'Favorite Product'})

    # Last Product: Product from the customer's last successful deal
    last_product = successful_deals.sort_values('تاریخ انجام معامله').groupby('کد دیدار شخص معامله').tail(1)
    last_product = last_product[['کد دیدار شخص معامله', 'عنوان محصول']].rename(columns={'عنوان محصول': 'Last Product'})

    # Merge Favorite Product and Last Product into rfm_data
    rfm_data = rfm_data.merge(favorite_product, left_on='Customer ID', right_on='کد دیدار شخص معامله', how='left')
    rfm_data = rfm_data.merge(last_product, left_on='Customer ID', right_on='کد دیدار شخص معامله', how='left')

    # Drop the extra 'کد دیدار شخص معامله' columns
    rfm_data.drop(columns=['کد دیدار شخص معامله_x', 'کد دیدار شخص معامله_y'], inplace=True)

    # -------------------------------------------------------------------------------------------

    return rfm_data

# Function for RFM segmentation
@st.cache_data
def rfm_segmentation(data):
    data = data[(data['Monetary'] > 0) & (data['Customer ID'] != 0)]
    # Define R, F, M thresholds based on quantiles to categorize scores
    buckets = data[['Recency', 'Frequency', 'Monetary']].quantile([1/3, 2/3]).to_dict()

    # Define the RFM segmentation function
    def rfm_segment(row):
        # Recency scoring
        if row['Recency'] >= 296:
            r_score = 1
        elif row['Recency'] >= 185:
            r_score = 2
        elif row['Recency'] >= 76:
            r_score = 3
        else:
            r_score = 4

        # Frequency scoring based on quantiles
        if row['Frequency'] <= buckets['Frequency'][1/3]:
            f_score = 1
        elif row['Frequency'] <= buckets['Frequency'][2/3]:
            f_score = 2
        else:
            f_score = 3

        # Monetary scoring based on quantiles
        if row['Monetary'] <= buckets['Monetary'][1/3]:
            m_score = 1
        elif row['Monetary'] <= buckets['Monetary'][2/3]:
            m_score = 2
        else:
            m_score = 3

        return f"{r_score}{f_score}{m_score}"

    # Apply the segmentation function to categorize customers into RFM segments
    data['RFM_segment'] = data.apply(rfm_segment, axis=1)

    # Define segment labels based on RFM combinations
    segment_labels = {
        '111': 'Churned',
        '112': 'Churned',
        '113': 'Lost Big Spenders',
        '121': 'Churned',
        '122': 'Churned',
        '123': 'Lost Big Spenders',
        '131': 'Hibernating',
        '132': 'Big Loss',
        '133': 'Big Loss',
        '211': 'Low Value',
        '212': 'At Risk',
        '213': 'At Risk',
        '221': 'Low Value',
        '222': 'At Risk',
        '223': 'At Risk',
        '231': 'At Risk',
        '232': 'At Risk',
        '233': 'At Risk',
        '311': 'Low Value',
        '312': 'Promising',
        '313': 'Big Spenders',
        '321': 'Promising',
        '322': 'Promising',
        '323': 'Promising',
        '331': 'Loyal Customers',
        '332': 'Loyal Customers',
        '333': 'Loyal Customers',
        '411': 'Promising',
        '412': 'Promising',
        '413': 'Big Spenders',
        '421': 'Price Sensitive',
        '422': 'Loyal Customers',
        '423': 'Loyal Customers',
        '431': 'Price Sensitive',
        '432': 'Loyal Customers',
        '433': 'Champions'
    }

    # Map the segment label to each RFM segment
    data['RFM_segment_label'] = data['RFM_segment'].map(segment_labels)
    return data

# Function to normalize RFM values for plotting
@st.cache_data
def normalize_rfm(data):
    scaler = MinMaxScaler()

    # For Recency, invert the scale so that higher is better (more recent purchase)
    data['Recency_norm'] = scaler.fit_transform(data[['Recency']])
    data['Recency_norm'] = 1 - data['Recency_norm']  # Invert Recency scores

    # Normalize Frequency and Monetary normally
    data[['Frequency_norm', 'Monetary_norm']] = scaler.fit_transform(
        data[['Frequency', 'Monetary']]
    )
    return data

# Global functions for data conversion (moved outside conditional blocks)
@st.cache_data
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

def main():
    # Set page config
    st.set_page_config(
        page_title="داشبورد سگمنت‌بندی مشتریان تهران‌مبله",
        page_icon="📊",
        layout="wide",
    )

    # Title
    st.title("Customer Segmentation Dashboard - Tehran Moble")

    # File uploader
    st.sidebar.header("Upload your deals Excel file")
    uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        try:
            # Load and preprocess data
            data_load_state = st.text('Loading and processing data...')
            data = load_data(uploaded_file)

            # Get unique options for filters
            product_options = data['عنوان محصول'].dropna().unique().tolist()
            product_options.sort()

            sellers_options = data['مسئول معامله'].dropna().unique().tolist()
            sellers_options.sort()

            sale_channels_options = data['شیوه آشنایی معامله'].dropna().unique().tolist()
            sale_channels_options.sort()

            vip_options = data['VIP Status'].dropna().unique().tolist()
            vip_options.sort()

        
            
            # ------------------ Navigation ------------------
            st.sidebar.header("Navigation")
            page = st.sidebar.radio("Go to", ['General', 'Churned Analysis','Compare RFM Segments Over Time', 'Portfolio Analysis', 'Seller Analysis', 'Sale Channel Analysis', 'VIP Analysis','Customer Batch Edit', 'Customer Inquiry Module','Arrival Analysis','Price Elasticity Analysis'])

            filtered_data = data.copy()


            # Cache the filtered data
            @st.cache_data
            def get_filtered_data():
                return filtered_data.copy()

            filtered_data = get_filtered_data()

            # Ensure 'تاریخ انجام معامله' is datetime and handle NaT
            filtered_data['تاریخ انجام معامله'] = pd.to_datetime(filtered_data['تاریخ انجام معامله'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['تاریخ انجام معامله'])

            # Calculate RFM (Current RFM) on entire data
            rfm_data = calculate_rfm(data)
            rfm_data = rfm_segmentation(rfm_data)
            rfm_data = normalize_rfm(rfm_data)
            data_load_state.text('Loading and processing data...done!')

            stay_options = rfm_data['Is Monthly'].dropna().unique().tolist()
            stay_options.sort()

            current_status_options=rfm_data['Is staying'].dropna().unique().tolist()
            current_status_options.sort()

            # Define colors for segments (used globally)
            COLOR_MAP = {
                "Champions": "#00CC96",            # Green
                "Loyal Customers": "#19D3F3",      # Light Blue
                "Promising": "#B6E880",            # Light Green
                "Big Spenders": "#FF6692",         # Pink
                "Price Sensitive": "#FFA15A",      # Orange
                "At Risk": "#AB63FA",              # Purple
                "Churned": "#c21e56",              # Red
                "Hibernating": "#636EFA",          # Blue
                "Lost Big Spenders": "#FF7415",    # Orange-Red
                "Big Loss": "#cdca49",             # Olive/Khaki
                "Low Value": "#D3D3D3",            # Gray
            }

            # Filter RFM data based on customers in filtered_data
            customers_in_filtered_data = filtered_data['کد دیدار شخص معامله'].unique()
            rfm_data_filtered_global = rfm_data[rfm_data['Customer ID'].isin(customers_in_filtered_data)]

            # ------------------ Pages ------------------


###
###############################################################################
# REPLACEMENT CODE FOR THE 'General' PAGE ONLY
###############################################################################
            if page == 'General':

                # -- 1) Prepare an empty DataFrame for potential filtered data
                rfm_data_filtered_plots = pd.DataFrame()

                # -- 2) VIP Filter
                vip_options_page = sorted(rfm_data_filtered_global['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_plots')

                if select_all_vips_page:
                    selected_vips_plots = vip_options_page
                else:
                    selected_vips_plots = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],  # empty if user doesn’t pick
                        key='vips_multiselect_plots'
                    )

                if not select_all_vips_page and not selected_vips_plots:
                    # If user unchecks and picks nothing, default all
                    selected_vips_plots = vip_options_page

                rfm_data_filtered_global = rfm_data_filtered_global[rfm_data_filtered_global['VIP Status'].isin(selected_vips_plots)]

                # -- 3) Blacklist Filter
                if 'BlackList Status' not in data.columns:
                    data['BlackList Status'] = extract_blacklist_status(data['نام خانوادگی شخص معامله'])

                blacklist_options_page = sorted(data['BlackList Status'].unique())
                select_all_blacklist_page = st.checkbox("Select all Black List statuses", value=True, key='select_all_blacklist_portfolio')

                if select_all_blacklist_page:
                    selected_blacklist_page = blacklist_options_page
                else:
                    selected_blacklist_page = st.multiselect(
                        "Select Black List Status:",
                        options=blacklist_options_page,
                        default=[],
                        key='blacklist_multiselect_portfolio'
                    )

                if not select_all_blacklist_page and not selected_blacklist_page:
                    selected_blacklist_page = blacklist_options_page

                # Filter original data by blacklist, then filter rfm_data accordingly
                data_for_general = data[data['BlackList Status'].isin(selected_blacklist_page)]
                rfm_data_filtered_global = rfm_data_filtered_global[
                    rfm_data_filtered_global['Customer ID'].isin(data_for_general['کد دیدار شخص معامله'])
                ]

                # -- 4) Segment Filter
                segment_options = sorted(rfm_data_filtered_global['RFM_segment_label'].unique())
                select_all_segments = st.checkbox("Select all segments", value=True, key='select_all_segments_plots')

                if select_all_segments:
                    selected_segments_plots = segment_options
                else:
                    selected_segments_plots = st.multiselect(
                        "Select RFM Segments:",
                        options=segment_options,
                        default=[],
                        key='segments_multiselect_plots'
                    )

                if not select_all_segments and not selected_segments_plots:
                    selected_segments_plots = segment_options

                if selected_segments_plots:
                    rfm_data_filtered_plots = rfm_data_filtered_global[
                        rfm_data_filtered_global['RFM_segment_label'].isin(selected_segments_plots)
                    ]
                else:
                    st.warning("No segments selected. Please select at least one segment.")

                if rfm_data_filtered_plots.empty:
                    st.warning("No data available for the selected segments/VIP/Blacklist filters.")
                else:
                    # Create 4 tabs
                    tab4, tab1, tab2, tab3 = st.tabs(["Customer Segmentation Data", "Pie Chart", "3D Scatter Plot", "Histograms"])

                    # ------------------------- Tab 1: Pie Chart -------------------------
                    with tab1:
                        st.subheader("Distribution of RFM Segments")
                        rfm_segment_counts = rfm_data_filtered_plots['RFM_segment_label'].value_counts().reset_index()
                        rfm_segment_counts.columns = ['RFM_segment_label', 'Count']

                        fig_pie = px.pie(
                            rfm_segment_counts,
                            names='RFM_segment_label',
                            values='Count',
                            color='RFM_segment_label',
                            color_discrete_map=COLOR_MAP,
                            hole=0.4
                        )
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie)

                    # ------------------------- Tab 2: 3D Scatter Plot -------------------
                    with tab2:
                        st.subheader("3D Scatter Plot of RFM Segments")
                        fig_3d = px.scatter_3d(
                            rfm_data_filtered_plots,
                            x='Recency_norm',
                            y='Frequency_norm',
                            z='Monetary_norm',
                            color='RFM_segment_label',
                            color_discrete_map=COLOR_MAP,
                            hover_data=['Customer ID', 'First Name', 'Last Name', 'VIP Status'],
                            title='RFM Segments (Normalized Space)'
                        )
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title='Recency (Higher=Better)',
                                yaxis_title='Frequency',
                                zaxis_title='Monetary'
                            ),
                            legend_title='RFM Segments'
                        )
                        st.plotly_chart(fig_3d)

                    # ------------------------- Tab 3: Histograms -------------------------
                    with tab3:
                        st.subheader("RFM Metrics Distribution")

                        # Recency Histogram
                        fig_recency = px.histogram(
                            rfm_data_filtered_plots,
                            x='Recency',
                            nbins=50,
                            title='Recency Distribution',
                            color='RFM_segment_label',
                            color_discrete_map=COLOR_MAP
                        )
                        st.plotly_chart(fig_recency)

                        # Frequency Histogram
                        fig_frequency = px.histogram(
                            rfm_data_filtered_plots,
                            x='Frequency',
                            nbins=50,
                            title='Frequency Distribution',
                            color='RFM_segment_label',
                            color_discrete_map=COLOR_MAP
                        )
                        st.plotly_chart(fig_frequency)

                        # Monetary Histogram
                        fig_monetary = px.histogram(
                            rfm_data_filtered_plots,
                            x='Monetary',
                            nbins=50,
                            title='Monetary Value Distribution',
                            labels={'Monetary': 'Monetary Value'},
                            color='RFM_segment_label',
                            color_discrete_map=COLOR_MAP
                        )
                        st.plotly_chart(fig_monetary)

                    # ------------------ Tab 4: Customer Segmentation Data ------------------
                    with tab4:
                        st.subheader("Customer Segmentation Data")

                        @st.cache_data
                        def get_filter_options(data, rfm_data):
                            product_options = sorted(data['عنوان محصول'].dropna().unique().tolist())
                            stay_options = sorted(rfm_data['Is Monthly'].dropna().unique().tolist())
                            current_status_options = sorted(rfm_data['Is staying'].dropna().unique().tolist())
                            return product_options, stay_options, current_status_options

                        product_options, stay_options, current_status_options = get_filter_options(data, rfm_data)

                        # ~~~~~~~~~~~~~ Additional function to add new metrics ~~~~~~~~~~~~~
                        @st.cache_data
                        def add_additional_metrics(df_in):
                            """
                            Demonstration placeholder for:
                            - Probability_of_Churn (via a mock logistic regression or formula)
                            - Customer_Lifespan
                            - CLTV
                            - CRR (1 - churn probability for illustration)
                            In real usage, you'd train a model on actual churn labels.
                            """

                            # If empty or missing columns, return as-is
                            if df_in.empty or 'Recency' not in df_in.columns:
                                return df_in

                            # Copy to avoid mutating original
                            df = df_in.copy()

                            # ~~~ STEP 1: Create a toy churn probability ~~~
                            # For demonstration: Probability_of_Churn = logistic function of (Recency, Frequency, Monetary)
                            # In a real scenario, you’d load a trained model or actually train one offline.
                            import numpy as np
                            import pandas as pd
                            from sklearn.linear_model import LogisticRegression

                            # We'll do a quick synthetic approach:
                            #   - Generate a mock binary label based on Recency>200 => "churned"
                            #   - Then train a logistic model for demonstration
                            df['churn_label'] = np.where(df['Recency'] > 200, 1, 0)

                            # Prepare features
                            X = df[['Recency', 'Frequency', 'Monetary']].fillna(0)
                            y = df['churn_label']

                            if len(X['Recency'].unique()) > 1:
                                # Fit a simple logistic regression
                                model = LogisticRegression()
                                model.fit(X, y)
                                churn_probs = model.predict_proba(X)[:, 1]
                            else:
                                # If no variance in Recency or data is too small, fallback
                                churn_probs = np.repeat(0.5, len(df))

                            df['Probability_of_Churn'] = churn_probs

                            # ~~~ STEP 2: Define Customer_Lifespan ~~~
                            # A naive approach: we’ll define it as 1 / (churn_prob), clipped.
                            df['Customer_Lifespan'] = np.where(
                                df['Probability_of_Churn'] < 0.01,
                                570,  # big cap for near-zero churn prob
                                1.0 / df['Probability_of_Churn']
                            )

                            # ~~~ STEP 3: CLTV ~~~
                            # For demonstration: CLTV = Monetary * Frequency * Customer_Lifespan
                            df['CLTV'] = df['Monetary'] * df['Frequency'] * df['Customer_Lifespan']

                            # ~~~ STEP 4: CRR ~~~
                            # Another naive approach: CRR = 1 - Probability_of_Churn
                            df['CRR'] = 1 - df['Probability_of_Churn']

                            # drop the synthetic churn label
                            df.drop(columns=['churn_label'], inplace=True, errors='ignore')

                            return df

                        # The DataFrame we’ll display in the table:
                        rfm_data_filtered_table = rfm_data_filtered_global.copy()

                        # ------------------ Product Filter ------------------
                        st.subheader("Filter Table by Products")
                        select_all_products_table = st.checkbox("Select all products", value=True, key='select_all_products_table')

                        if select_all_products_table:
                            selected_products_table = product_options
                        else:
                            selected_products_table = st.multiselect(
                                "Select products (عنوان محصول):",
                                options=product_options,
                                default=[],
                                key='products_multiselect_table'
                            )

                        if selected_products_table:
                            cust_ids_with_products = data[data['عنوان محصول'].isin(selected_products_table)]['کد دیدار شخص معامله'].unique()
                            rfm_data_filtered_table = rfm_data_filtered_table[rfm_data_filtered_table['Customer ID'].isin(cust_ids_with_products)]
                        else:
                            st.warning("No products selected. Displaying all products.")

                        # ------------------ "Monthly" Filter (Is Monthly) ------------------
                        min_nights = st.number_input(
                            "Enter minimum number of nights for monthly guests:",
                            min_value=0, value=15, step=1, key='min_nights_filter'
                        )
                        # Recompute 'Is Monthly' with chosen threshold
                        rfm_data_filtered_table['Is Monthly'] = (
                            (rfm_data_filtered_table['Total Nights'] / rfm_data_filtered_table['Frequency']).fillna(0) >= min_nights
                        )

                        select_all_staying_table = st.checkbox(
                            "Select all guest types (Monthly or not)", 
                            value=True, 
                            key='select_all_staying_table'
                        )

                        if select_all_staying_table:
                            selected_staying_table = [True, False]  # since 'Is Monthly' is boolean
                        else:
                            # user picks among True or False
                            staying_options_label = ["Monthly Guests","Non-Monthly Guests"]
                            selected_bool_values = st.multiselect(
                                "Select guest type (monthly or not):",
                                options=staying_options_label,
                                default=[]
                            )
                            # convert to booleans
                            mapping = {"Monthly Guests": True, "Non-Monthly Guests": False}
                            selected_staying_table = [mapping[val] for val in selected_bool_values]

                        # If user picks nothing => show all
                        if not selected_staying_table:
                            selected_staying_table = [True, False]

                        rfm_data_filtered_table = rfm_data_filtered_table[rfm_data_filtered_table['Is Monthly'].isin(selected_staying_table)]

                        # ------------------ "Is staying" Filter ------------------
                        select_all_current_status_table = st.checkbox(
                            "Select all current status (currently staying or not)",
                            value=True,
                            key='select_all_current_status_table'
                        )

                        if select_all_current_status_table:
                            selected_current_status_table = [True, False]
                        else:
                            # user picks among True or False
                            status_options_label = ["Currently Staying","Not Staying"]
                            chosen_status = st.multiselect(
                                "Select current status (currently staying or not):",
                                options=status_options_label,
                                default=[]
                            )
                            mapping_status = {"Currently Staying": True, "Not Staying": False}
                            selected_current_status_table = [mapping_status[val] for val in chosen_status]

                        if not selected_current_status_table:
                            selected_current_status_table = [True, False]

                        rfm_data_filtered_table = rfm_data_filtered_table[
                            rfm_data_filtered_table['Is staying'].isin(selected_current_status_table)
                        ]

                        # ~~~~~~~~~~~~~ Add the additional metrics columns here ~~~~~~~~~~~~~
                        rfm_data_filtered_table = add_additional_metrics(rfm_data_filtered_table)

                        # Show final table
                        st.write(rfm_data_filtered_table[[
                            'Customer ID', 'First Name', 'Last Name', 'VIP Status', 'Phone Number',
                            'Recency', 'Frequency', 'Monetary', 'average stay', 'Is Monthly', 
                            'Is staying', 'Favorite Product', 'Last Product', 'RFM_segment_label',
                            # New columns:
                            'Probability_of_Churn', 'Customer_Lifespan', 'CLTV', 'CRR'
                        ]])

                        # Download buttons
                        from io import BytesIO

                        csv_data = convert_df(rfm_data_filtered_table)
                        excel_data = convert_df_to_excel(rfm_data_filtered_table)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download data as CSV",
                                data=csv_data,
                                file_name='rfm_segmentation_with_churn.csv',
                                mime='text/csv',
                            )
                        with col2:
                            st.download_button(
                                label="Download data as Excel",
                                data=excel_data,
                                file_name='rfm_segmentation_with_churn.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            )

                        # Done with Tab 4 in 'General' page
            ###############################################################################

            elif page == 'Price Elasticity Analysis':
                price_elasticity_page(data)
            elif page == 'Churned Analysis':
                churned_analysis_page(rfm_data) 
            elif page == 'Compare RFM Segments Over Time':


                # ------------------ Compare RFM Segments Over Time ------------------

                st.subheader("Compare RFM Segments Over Time")

                # VIP Filter for this page
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses for comparison", value=True, key='select_all_vips_comparison')

                if select_all_vips_page:
                    selected_vips_comparison = vip_options_page
                else:
                    selected_vips_comparison = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_comparison'
                    )

                # Use a form to prevent automatic reruns
                with st.form(key='comparison_form'):
                    # Date Input
                    comparison_date = st.date_input("Select a date for comparison", value=datetime.today())

                    # Ensure that the date is not in the future
                    if comparison_date > datetime.today().date():
                        st.error("The comparison date cannot be in the future.")
                        submit_button = st.form_submit_button(label='Submit')
                    else:
                        # Get list of unique segments
                        segment_options = ['All'] + sorted(rfm_data['RFM_segment_label'].dropna().unique())

                        col1, col2 = st.columns(2)
                        with col1:
                            from_segment = st.selectbox("Select 'FROM' Segment (Before)", options=segment_options)
                        with col2:
                            to_segment = st.selectbox("Select 'TO' Segment (After)", options=segment_options)

                        # Show Results button
                        submit_button = st.form_submit_button(label='Show Results')

                if 'submit_button' in locals() and submit_button:
                    # Filter data before the selected date
                    data_before_date = data[data['تاریخ انجام معامله'] <= pd.to_datetime(comparison_date)]

                    if data_before_date.empty:
                        st.warning("No data available before the selected date.")
                    else:
                        # Calculate RFM1 (RFM before the selected date)
                        rfm_data1 = calculate_rfm(data_before_date, today=comparison_date)
                        rfm_data1 = rfm_segmentation(rfm_data1)

                        # Filter RFM data based on selected VIP statuses
                        rfm_data1 = rfm_data1[rfm_data1['VIP Status'].isin(selected_vips_comparison)]
                        rfm_data_filtered = rfm_data[rfm_data['VIP Status'].isin(selected_vips_comparison)]

                        # Prepare data for comparison
                        # Merge RFM1 and RFM2 on 'Customer ID'
                        comparison_df = rfm_data1[['Customer ID', 'First Name', 'Last Name', 'Phone Number', 'VIP Status', 'RFM_segment_label']].merge(
                            rfm_data_filtered[['Customer ID','average stay','Is Monthly','Is staying', 'RFM_segment_label']],
                            on='Customer ID',
                            how='inner',
                            suffixes=('_RFM1', '_RFM2')
                        )

                        # Handle the cases
                        if from_segment == 'All' and to_segment == 'All':
                            st.error("Please select at least one segment in 'FROM' or 'TO'.")
                        else:
                            if from_segment != 'All':
                                comparison_df = comparison_df[comparison_df['RFM_segment_label_RFM1'] == from_segment]
                            if to_segment != 'All':
                                comparison_df = comparison_df[comparison_df['RFM_segment_label_RFM2'] == to_segment]

                            if comparison_df.empty:
                                st.warning("No customers found for the selected segment transitions and VIP statuses.")
                            else:
                                # Display count and bar chart
                                if from_segment!='All':
                                    counts = comparison_df['RFM_segment_label_RFM2'].value_counts().reset_index()
                                    counts.columns = ['RFM_segment_label_RFM2', 'Count']
                                elif to_segment!='All':
                                    counts = comparison_df['RFM_segment_label_RFM1'].value_counts().reset_index()
                                    counts.columns = ['RFM_segment_label_RFM1', 'Count']

                                st.write(f"Number of customers matching the criteria: **{len(comparison_df)}**")

                                if from_segment!='All':
                                    fig = px.bar(
                                        counts,
                                        x='RFM_segment_label_RFM2',
                                        y='Count',
                                        color='RFM_segment_label_RFM2',
                                        color_discrete_map=COLOR_MAP,
                                        text='Count',
                                        labels={'RFM_segment_label_RFM2': 'Segment After that date', 'Count': 'Number of Customers'}
                                    )
                                elif to_segment!='All':
                                    fig = px.bar(
                                        counts,
                                        x='RFM_segment_label_RFM1',
                                        y='Count',
                                        color='RFM_segment_label_RFM1',
                                        color_discrete_map=COLOR_MAP,
                                        text='Count',
                                        labels={'RFM_segment_label_RFM1': 'Segment Before that date', 'Count': 'Number of Customers'}
                                    )
                                
                                if to_segment=='All' or from_segment=='All':
                                    fig.update_traces(textposition='outside')
                                    st.plotly_chart(fig)

                                # Show customer table
                                st.subheader("Customer Details")
                                customer_table = comparison_df[['Customer ID', 'First Name', 'Last Name', 'Phone Number', 'VIP Status','average stay','Is Monthly','Is staying', 'RFM_segment_label_RFM1', 'RFM_segment_label_RFM2']]
                                customer_table.rename(columns={
                                    'RFM_segment_label_RFM1': 'Before Segment',
                                    'RFM_segment_label_RFM2': 'After Segment'
                                }, inplace=True)
                                st.write(customer_table)

                                # Download buttons
                                csv_data = convert_df(customer_table)
                                excel_data = convert_df_to_excel(customer_table)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download data as CSV",
                                        data=csv_data,
                                        file_name='rfm_segment_comparison.csv',
                                        mime='text/csv',
                                    )
                                with col2:
                                    st.download_button(
                                        label="Download data as Excel",
                                        data=excel_data,
                                        file_name='rfm_segment_comparison.xlsx',
                                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                    )

            elif page == 'Portfolio Analysis':
                # ------------------ Portfolio Analysis ------------------

                st.subheader("Portfolio Analysis by Cluster and Product")

                # Get unique clusters from RFM data
                cluster_options = rfm_data['RFM_segment_label'].unique().tolist()
                cluster_options.sort()
                select_all_clusters = st.checkbox("Select all clusters", value=True, key='select_all_clusters_portfolio')

                if select_all_clusters:
                    selected_clusters = cluster_options
                else:
                    selected_clusters = st.multiselect(
                        "Select Clusters:",
                        options=cluster_options,
                        default=[],
                        key='clusters_multiselect_portfolio'
                    )

                # Filter by Complex
                complex_options = data['Complex'].dropna().unique().tolist()
                complex_options.sort()
                select_all_complex = st.checkbox("Select all complexes", value=True, key='select_all_complex')

                if select_all_complex:
                    selected_complexes = complex_options
                else:
                    selected_complexes = st.multiselect(
                        "Select Complexes:",
                        options=complex_options,
                        default=[],
                        key='complexes_multiselect'
                    )

                data_filtered_by_complex = data[data['Complex'].isin(selected_complexes)]

                # Filter by Type
                type_options = data_filtered_by_complex['عنوان محصول'].dropna().unique().tolist()
                type_options.sort()
                select_all_types = st.checkbox("Select all types", value=True, key='select_all_types')

                if select_all_types:
                    selected_types = type_options
                else:
                    selected_types = st.multiselect(
                        "Select Types:",
                        options=type_options,
                        default=[],
                        key='types_multiselect'
                    )

                data_filtered_by_type = data_filtered_by_complex[data_filtered_by_complex['عنوان محصول'].isin(selected_types)]

                # Filter by Blacklist Status
                blacklist_options = sorted(data['BlackList Status'].unique())
                select_all_blacklist = st.checkbox("Select all BlackList statuses", value=True, key='select_all_blacklist')

                if select_all_blacklist:
                    selected_blacklist = blacklist_options
                else:
                    selected_blacklist = st.multiselect(
                        "Select BlackList Status:",
                        options=blacklist_options,
                        default=[],
                        key='blacklist_multiselect'
                    )

                data_filtered_by_blacklist = data_filtered_by_type[data_filtered_by_type['BlackList Status'].isin(selected_blacklist)]

                # VIP Filter
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_portfolio')

                if select_all_vips_page:
                    selected_vips_portfolio = vip_options_page
                else:
                    selected_vips_portfolio = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=[],
                        key='vips_multiselect_portfolio'
                    )

                # Apply filters
                with st.form(key='portfolio_form'):
                    apply_portfolio = st.form_submit_button(label='Apply')

                if apply_portfolio:
                    if not selected_clusters:
                        st.warning("Please select at least one cluster.")
                    elif not selected_vips_portfolio:
                        st.warning("Please select at least one VIP status.")
                    else:
                        # Get customers in selected clusters and VIP statuses
                        customers_in_clusters = rfm_data[(rfm_data['RFM_segment_label'].isin(selected_clusters)) &
                                                        (rfm_data['VIP Status'].isin(selected_vips_portfolio))]['Customer ID'].unique()

                        # Filter deals data
                        deals_filtered = data_filtered_by_blacklist[data_filtered_by_blacklist['کد دیدار شخص معامله'].isin(customers_in_clusters)]

                        if deals_filtered.empty:
                            st.warning("No deals found for the selected clusters, VIP statuses, and products.")
                        else:
                            # Frequency distribution
                            frequency_distribution = deals_filtered.groupby('عنوان محصول').size().reset_index(name='Frequency')

                            # Monetary distribution
                            monetary_distribution = deals_filtered.groupby('عنوان محصول')['ارزش معامله'].sum().reset_index()

                            # Plot Frequency Distribution
                            st.subheader("Frequency Distribution of Products")
                            fig_freq = px.bar(
                                frequency_distribution,
                                x='عنوان محصول',
                                y='Frequency',
                                title='Frequency Distribution',
                                labels={'عنوان محصول': 'Product', 'Frequency': 'Number of Purchases'},
                                text='Frequency'
                            )
                            fig_freq.update_traces(textposition='outside')
                            st.plotly_chart(fig_freq)

                            # Plot Monetary Distribution
                            st.subheader("Monetary Distribution of Products")
                            fig_monetary = px.bar(
                                monetary_distribution,
                                x='عنوان محصول',
                                y='ارزش معامله',
                                title='Monetary Distribution',
                                labels={'عنوان محصول': 'Product', 'ارزش معامله': 'Total Monetary Value'},
                                text='ارزش معامله'
                            )
                            fig_monetary.update_traces(textposition='outside')
                            st.plotly_chart(fig_monetary)

                            # Customer Details Table
                            st.subheader("Customer Details")
                            successful_deals = deals_filtered[deals_filtered['وضعیت معامله'] == 'موفق']
                            customer_nights = successful_deals.groupby(['کد دیدار شخص معامله', 'عنوان محصول'])['تعداد شب'].sum().unstack(fill_value=0)

                            customer_details = rfm_data[rfm_data['Customer ID'].isin(customers_in_clusters)][['Customer ID', 'First Name', 'Last Name', 'VIP Status','average stay','Is Monthly','Is staying', 'RFM_segment_label', 'Recency', 'Frequency', 'Monetary']]
                            customer_details = customer_details.merge(customer_nights, left_on='Customer ID', right_index=True, how='inner').fillna(0)

                            st.write(customer_details)

                            # Download buttons
                            csv_data = convert_df(customer_details)
                            excel_data = convert_df_to_excel(customer_details)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Download data as CSV",
                                    data=csv_data,
                                    file_name='portfolio_analysis.csv',
                                    mime='text/csv',
                                )
                            with col2:
                                st.download_button(
                                    label="Download data as Excel",
                                    data=excel_data,
                                    file_name='portfolio_analysis.xlsx',
                                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                )

            elif page == 'Customer Batch Edit':

                    st.title("Customer Batch Edit")

                    st.write("""
                    This tool allows you to upload a list of contacts, specify a word to add or remove from their last names, and perform batch updates.
                    """)

                    # File uploader for the Excel file
                    uploaded_file = st.file_uploader("Upload Contacts File (Excel)", type=["xlsx"])

                    # Input fields for the word and action
                    preset_word = st.text_input("Enter the word to add/remove")
                    action = st.selectbox("Choose an action", options=["Select", "Add", "Remove"])
                    password = st.text_input("Enter  to confirm action", type="")

                    if st.button("Execute"):

                        # Validate 
                        if password != st.secrets["change_password"]:
                            st.error("Invalid password. Please try again.")
                        elif uploaded_file is None:
                            st.error("Please upload a valid Excel file.")
                        elif action not in {"Add", "Remove"}:
                            st.error("Please select a valid action (Add or Remove).")
                        elif not preset_word.strip():
                            st.error("The word to add/remove cannot be empty.")
                        else:
                            # Load phone numbers from the uploaded Excel file
                            try:
                                phone_numbers = pd.read_excel(uploaded_file, usecols=[0], header=None).squeeze().tolist()
                            except Exception as e:
                                st.error("Failed to read the uploaded Excel file. Please ensure it has phone numbers in the first column.")
                                st.error(str(e))
                                st.stop()

                            # Initialize success and error counts
                            success_count = 0
                            error_count = 0

                            # Process each phone number
                            for mobile_phone in phone_numbers:
                                # Endpoint for searching contacts
                                search_endpoint = '/contact/personsearch'
                                search_url = f"https://app.didar.me/api{search_endpoint}?apikey=uvio38zfgpbbsasyn0f8pl61b4ve6va3"

                                # Search payload
                                search_payload = {
                                    "Criteria": {
                                        "IsDeleted": 0,
                                        "IsPinned": -1,
                                        "IsVIP": -1,
                                        "LeadType": -1,
                                        "Pin": -1,
                                        "SortOrder": 1,
                                        "Keywords": str(mobile_phone),
                                        "OwnerId": "00000000-0000-0000-0000-000000000000",
                                        "SearchFromTime": "1930-01-01T00:00:00.000Z",
                                        "SearchToTime": "9999-12-01T00:00:00.000Z",
                                        "CustomFields": [],
                                        "FilterId": None
                                    },
                                    "From": 0,
                                    "Limit": 30
                                }

                                # Headers
                                headers = {
                                    'Content-Type': 'application/json'
                                }

                                # Search for the contact
                                response = requests.post(search_url, headers=headers, json=search_payload)

                                if response.status_code == 200:
                                    response_data = response.json()
                                    contacts = response_data.get('Response', {}).get('List', [])

                                    if contacts:
                                        # Process the first contact found
                                        contact = contacts[0]
                                        last_name = contact.get('LastName', '')

                                        if action == 'Add':
                                            # Add the preset word to the last name
                                            updated_last_name = last_name + " " + preset_word
                                        elif action == 'Remove':
                                            # Remove the preset word from the last name
                                            pattern = r'\s*' + re.escape(preset_word) + r'$'
                                            updated_last_name = re.sub(pattern, '', last_name)

                                        if updated_last_name != last_name:
                                            # Update contact details
                                            contact['LastName'] = updated_last_name
                                            contact['DisplayName'] = (contact.get('FirstName', '') + ' ' + updated_last_name).strip()

                                            # Remove unnecessary fields
                                            fields_to_remove = [
                                                'CanDelete', 'CanEdit', 'IsMine', 'HasAccess', '_Type', 'OwnerId_Old', 
                                                'Segments', 'Owner', 'ContactStatus', 'KeepInTouch', 'Fields'
                                            ]
                                            for field in fields_to_remove:
                                                contact.pop(field, None)

                                            # Handle segments
                                            segments = contact.get('Segments', [])
                                            segment_ids = [segment.get('Id') for segment in segments]

                                            # Prepare save payload
                                            save_payload = {
                                                "Contact": contact,
                                                "SegmentIds": segment_ids
                                            }

                                            # Endpoint to save/update the contact
                                            save_endpoint = '/contact/save'
                                            save_url = f"https://app.didar.me/api{save_endpoint}?ApiKey=uvio38zfgpbbsasyn0f8pl61b4ve6va3"

                                            # Save the updated contact
                                            save_response = requests.post(save_url, headers=headers, json=save_payload)

                                            if save_response.status_code == 200:
                                                success_count += 1
                                            else:
                                                error_count += 1
                                        else:
                                            if action == 'Remove':
                                                st.warning(f"The word '{preset_word}' was not found in the last name of contact {mobile_phone}.")
                                            else:
                                                st.warning(f"Contact {mobile_phone} already has the word '{preset_word}' in the last name.")
                                    else:
                                        error_count += 1
                                        st.warning(f"No contact found for phone number {mobile_phone}.")
                                else:
                                    error_count += 1
                                    st.error(f"Failed to search for contact {mobile_phone}. Status code: {response.status_code}")

                            # Display summary of the operation
                            st.success(f"Batch operation completed: {success_count} succeeded, {error_count} failed.")


         
            elif page == 'Seller Analysis':
                st.subheader("Seller Analysis")

                # We use tabs for the four sections
                tabs = st.tabs(["Single Seller Analysis", "Compare Two Sellers", "Compare All Sellers", "RFM Sales Analysis"])

                @st.cache_data
                def get_first_successful_deal_date(df):
                    """Return a series mapping each customer to their first successful deal date."""
                    successful_deals_only = df[df['وضعیت معامله'] == 'موفق'].copy()
                    first_deal = successful_deals_only.groupby('کد دیدار شخص معامله')['تاریخ انجام معامله'].min()
                    return first_deal

                global_first_deal_date_series = get_first_successful_deal_date(data)

                ###########################################################################
                #  SINGLE SELLER ANALYSIS
                ###########################################################################
                with tabs[0]:
                    vip_options_page = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_seller_single')
                    if select_all_vips_page:
                        selected_vips_seller = vip_options_page
                    else:
                        selected_vips_seller = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_page,
                            default=[],
                            key='vips_multiselect_seller_single'
                        )

                    with st.form(key='seller_filters_form', clear_on_submit=False):
                        selected_seller = st.selectbox("Select a Seller:", options=sellers_options)
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input(
                            "Start Date", 
                            value=min_date,
                            min_value=min_date, 
                            max_value=max_date, 
                            key='seller_start_date_single'
                        )
                        end_date = st.date_input(
                            "End Date", 
                            value=max_date,
                            min_value=min_date, 
                            max_value=max_date, 
                            key='seller_end_date_single'
                        )

                        apply_seller_filters = st.form_submit_button(label='Apply Filters')

                    if "single_seller_data" not in st.session_state:
                        st.session_state.single_seller_data = None
                        st.session_state.single_seller_filtered_all = None
                        st.session_state.single_seller_kpi_df = None
                        st.session_state.single_seller_daily_df = None

                    if apply_seller_filters:
                        if selected_seller:
                            if selected_vips_seller:
                                date_filtered_data_all = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['مسئول معامله'] == selected_seller)
                                ]
                                date_filtered_data_all = date_filtered_data_all[date_filtered_data_all['VIP Status'].isin(selected_vips_seller)]
                                seller_data = date_filtered_data_all[date_filtered_data_all['وضعیت معامله'] == 'موفق']

                                if date_filtered_data_all.empty:
                                    st.warning("No deals found for this seller in the specified date range.")
                                    st.session_state.single_seller_data = None
                                    st.session_state.single_seller_filtered_all = None
                                    st.session_state.single_seller_kpi_df = None
                                    st.session_state.single_seller_daily_df = None
                                else:
                                    st.session_state.single_seller_filtered_all = date_filtered_data_all.copy()
                                    st.session_state.single_seller_data = seller_data.copy()

                                    total_deals = len(date_filtered_data_all)
                                    successful_deals_count = len(seller_data)
                                    success_rate = (successful_deals_count / total_deals)*100 if total_deals>0 else 0

                                    new_customers = 0
                                    returning_customers = 0
                                    if not seller_data.empty:
                                        unique_customers = seller_data['کد دیدار شخص معامله'].unique()
                                        for cid in unique_customers:
                                            first_deal_date = global_first_deal_date_series.get(cid, pd.NaT)
                                            if pd.isna(first_deal_date):
                                                continue
                                            if start_date <= first_deal_date.date() <= end_date:
                                                new_customers += 1
                                            elif first_deal_date.date() < start_date:
                                                returning_customers += 1

                                    avg_deal_value = seller_data['ارزش معامله'].mean() if not seller_data.empty else 0
                                    avg_nights = seller_data['تعداد شب'].mean() if not seller_data.empty else 0

                                    extentions = seller_data[seller_data['نوع خرید'] == 'تمدید']
                                    extentions_count = len(extentions)
                                    extention_rate = (extentions_count / successful_deals_count * 100) if successful_deals_count>0 else 0

                                    # Compare with previous period
                                    prev_period_length = (end_date - start_date).days + 1
                                    prev_end_date = start_date - timedelta(days=1)
                                    prev_start_date = prev_end_date - timedelta(days=prev_period_length - 1)
                                    prev_period_data_all = data[
                                        (data['تاریخ انجام معامله'] >= pd.to_datetime(prev_start_date)) &
                                        (data['تاریخ انجام معامله'] <= pd.to_datetime(prev_end_date)) &
                                        (data['مسئول معامله'] == selected_seller)
                                    ]
                                    prev_period_data_all = prev_period_data_all[prev_period_data_all['VIP Status'].isin(selected_vips_seller)]
                                    prev_seller_data = prev_period_data_all[prev_period_data_all['وضعیت معامله'] == 'موفق']

                                    if not prev_period_data_all.empty:
                                        prev_total_deals = len(prev_period_data_all)
                                        prev_successful_deals_count = len(prev_seller_data)
                                        prev_success_rate = (prev_successful_deals_count / prev_total_deals)*100 if prev_total_deals>0 else 0
                                        prev_avg_deal_value = prev_seller_data['ارزش معامله'].mean() if not prev_seller_data.empty else 0
                                        prev_avg_nights = prev_seller_data['تعداد شب'].mean() if not prev_seller_data.empty else 0
                                        prev_extentions = prev_seller_data[prev_seller_data['نوع خرید'] == 'تمدید']
                                        prev_extentions_count = len(prev_extentions)
                                        prev_extention_rate = (prev_extentions_count/prev_successful_deals_count*100) if prev_successful_deals_count>0 else 0

                                        prev_new_customers = 0
                                        prev_returning_customers = 0
                                        if not prev_seller_data.empty:
                                            unique_customers_prev = prev_seller_data['کد دیدار شخص معامله'].unique()
                                            for cid in unique_customers_prev:
                                                first_deal_date = global_first_deal_date_series.get(cid, pd.NaT)
                                                if pd.isna(first_deal_date):
                                                    continue
                                                if prev_start_date <= first_deal_date.date() <= prev_end_date:
                                                    prev_new_customers += 1
                                                elif first_deal_date.date() < prev_start_date:
                                                    prev_returning_customers += 1
                                    else:
                                        prev_total_deals = 0
                                        prev_successful_deals_count = 0
                                        prev_success_rate = 0
                                        prev_avg_deal_value = 0
                                        prev_avg_nights = 0
                                        prev_extention_rate = 0
                                        prev_new_customers = 0
                                        prev_returning_customers = 0

                                    st.session_state.single_seller_kpi_df = {
                                        'total_deals': total_deals,
                                        'successful_deals_count': successful_deals_count,
                                        'success_rate': success_rate,
                                        'avg_deal_value': avg_deal_value,
                                        'avg_nights': avg_nights,
                                        'extention_rate': extention_rate,
                                        'new_customers': new_customers,
                                        'returning_customers': returning_customers,
                                        'prev_total_deals': prev_total_deals,
                                        'prev_successful_deals_count': prev_successful_deals_count,
                                        'prev_success_rate': prev_success_rate,
                                        'prev_avg_deal_value': prev_avg_deal_value,
                                        'prev_avg_nights': prev_avg_nights,
                                        'prev_extention_rate': prev_extention_rate,
                                        'prev_new_customers': prev_new_customers,
                                        'prev_returning_customers': prev_returning_customers
                                    }

                                    # Build daily metrics
                                    daily_metrics = []
                                    days_range = pd.date_range(start=start_date, end=end_date, freq='D')
                                    earliest_global = global_first_deal_date_series.to_dict()
                                    for single_day in days_range:
                                        day_data_all = date_filtered_data_all[date_filtered_data_all['تاریخ انجام معامله'].dt.date == single_day.date()]
                                        day_data_success = day_data_all[day_data_all['وضعیت معامله'] == 'موفق']
                                        td = len(day_data_all)
                                        sd = len(day_data_success)
                                        dv = day_data_success['ارزش معامله'].mean() if not day_data_success.empty else 0
                                        nights_v = day_data_success['تعداد شب'].mean() if not day_data_success.empty else 0

                                        new_cus = 0
                                        ret_cus = 0
                                        if not day_data_success.empty:
                                            for ccid in day_data_success['کد دیدار شخص معامله'].unique():
                                                fdate = earliest_global.get(ccid, pd.NaT)
                                                if not pd.isna(fdate):
                                                    if single_day.date() == fdate.date():
                                                        new_cus += 1
                                                    elif fdate.date() < single_day.date():
                                                        ret_cus += 1

                                        daily_metrics.append({
                                            'Date': single_day,
                                            'Total Deals': td,
                                            'Successful Deals': sd,
                                            'New Customers': new_cus,
                                            'Returning Customers': ret_cus,
                                            'Average Deal Value': dv,
                                            'Average Nights': nights_v
                                        })
                                    daily_df = pd.DataFrame(daily_metrics)
                                    st.session_state.single_seller_daily_df = daily_df.copy()

                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select a seller.")

                    if (
                        st.session_state.single_seller_data is not None and
                        st.session_state.single_seller_filtered_all is not None and
                        st.session_state.single_seller_kpi_df is not None
                    ):
                        kpi_data = st.session_state.single_seller_kpi_df

                        total_deals = kpi_data['total_deals']
                        successful_deals_count = kpi_data['successful_deals_count']
                        success_rate = kpi_data['success_rate']
                        avg_deal_value = kpi_data['avg_deal_value']
                        avg_nights = kpi_data['avg_nights']
                        extention_rate = kpi_data['extention_rate']
                        new_customers = kpi_data['new_customers']
                        returning_customers = kpi_data['returning_customers']

                        prev_total_deals = kpi_data['prev_total_deals']
                        prev_successful_deals_count = kpi_data['prev_successful_deals_count']
                        prev_success_rate = kpi_data['prev_success_rate']
                        prev_avg_deal_value = kpi_data['prev_avg_deal_value']
                        prev_avg_nights = kpi_data['prev_avg_nights']
                        prev_extention_rate = kpi_data['prev_extention_rate']
                        prev_new_customers = kpi_data['prev_new_customers']
                        prev_returning_customers = kpi_data['prev_returning_customers']

                        def pct_diff(new_val, old_val):
                            if old_val == 0:
                                return None
                            return f"{((new_val - old_val)/abs(old_val)*100):.2f}%"

                        st.markdown("### Key Performance Indicators (KPIs)")
                        colKPI1, colKPI2, colKPI3, colKPI4 = st.columns(4)
                        colKPI1.metric(
                            "Total Deals",
                            f"{total_deals}",
                            pct_diff(total_deals, prev_total_deals)
                        )
                        colKPI2.metric(
                            "Successful Deals",
                            f"{successful_deals_count}",
                            pct_diff(successful_deals_count, prev_successful_deals_count)
                        )
                        colKPI3.metric(
                            "Success Rate (%)",
                            f"{success_rate:.2f}%",
                            pct_diff(success_rate, prev_success_rate)
                        )
                        colKPI4.metric(
                            "Avg. Deal Value",
                            f"{avg_deal_value:,.0f}",
                            pct_diff(avg_deal_value, prev_avg_deal_value)
                        )

                        colKPI5, colKPI6, colKPI7, colKPI8 = st.columns(4)
                        colKPI5.metric(
                            "New Customers",
                            f"{new_customers}",
                            pct_diff(new_customers, prev_new_customers)
                        )
                        colKPI6.metric(
                            "Returning Customers",
                            f"{returning_customers}",
                            pct_diff(returning_customers, prev_returning_customers)
                        )
                        colKPI7.metric(
                            "Avg. Nights",
                            f"{avg_nights:.2f}",
                            pct_diff(avg_nights, prev_avg_nights)
                        )
                        colKPI8.metric(
                            "Extention Rate",
                            f"{extention_rate:.2f}%",
                            pct_diff(extention_rate, prev_extention_rate)
                        )

                        st.write("---")

                        # Outlier Detection
                        st.markdown("**Outlier Detection in Deal Values**")
                        deals_df = st.session_state.single_seller_data[['ارزش معامله','تاریخ انجام معامله','کد دیدار شخص معامله','تعداد شب','نوع خرید','VIP Status']].copy()
                        deals_df.dropna(subset=['ارزش معامله'], inplace=True)
                        if len(deals_df) > 5:
                            q1, q3 = np.percentile(deals_df['ارزش معامله'], [25,75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = deals_df[(deals_df['ارزش معامله'] < lower_bound) | (deals_df['ارزش معامله'] > upper_bound)]
                            if not outliers.empty:
                                st.write(f"Detected {len(outliers)} outlier deal(s). Below is the table of those outlier deals:")
                                st.write(outliers)
                            else:
                                st.write("No outliers detected in deal values.")
                        else:
                            st.info("Not enough data to detect outliers reliably.")

                        seller_data = st.session_state.single_seller_data
                        if seller_data.empty:
                            st.warning("No successful deals found in this date range for the selected VIP statuses.")
                        else:
                            # RFM distribution if possible
                            if 'RFM_segment_label' not in rfm_data.columns:
                                st.warning("RFM_segment_label column not found in rfm_data. Can't show cluster distributions.")
                            else:
                                seller_customer_ids = seller_data['کد دیدار شخص معامله'].unique()
                                seller_rfm_data = rfm_data[rfm_data['Customer ID'].isin(seller_customer_ids)]
                                if seller_rfm_data.empty:
                                    st.warning("No RFM data available for the selected seller and VIP statuses.")
                                else:
                                    cluster_counts = seller_rfm_data['RFM_segment_label'].value_counts().reset_index()
                                    cluster_counts.columns = ['RFM_segment_label', 'Count']
                                    fig_seller_freq = px.bar(
                                        cluster_counts,
                                        x='RFM_segment_label',
                                        y='Count',
                                        title="Cluster Distribution (Frequency)",
                                        labels={'RFM_segment_label': 'RFM Segment','Count': 'Number of Customers'},
                                        text='Count',
                                        color='RFM_segment_label',
                                        color_discrete_sequence=px.colors.qualitative.Set1
                                    )
                                    fig_seller_freq.update_traces(textposition='outside')
                                    st.plotly_chart(fig_seller_freq)

                                    seller_monetary = seller_rfm_data.groupby('RFM_segment_label')['Monetary'].sum().reset_index()
                                    fig_seller_monetary = px.bar(
                                        seller_monetary,
                                        x='RFM_segment_label',
                                        y='Monetary',
                                        title="Cluster Distribution (Monetary)",
                                        labels={'RFM_segment_label': 'RFM Segment','Monetary': 'Total Monetary Value'},
                                        text='Monetary',
                                        color='RFM_segment_label',
                                        color_discrete_sequence=px.colors.qualitative.Set1
                                    )
                                    fig_seller_monetary.update_traces(textposition='outside')
                                    st.plotly_chart(fig_seller_monetary)

                            st.subheader("Customer Details")
                            customer_nights = seller_data.groupby('کد دیدار شخص معامله')['تعداد شب'].sum().reset_index()
                            customer_nights.rename(columns={'کد دیدار شخص معامله': 'Customer ID','تعداد شب': 'Total Nights'}, inplace=True)
                            if 'Customer ID' in rfm_data.columns:
                                customer_details = rfm_data[['Customer ID','First Name','Phone Number','Last Name','VIP Status','Recency','Frequency','Monetary','average stay','Is Monthly','Is staying']].copy()
                                if 'RFM_segment_label' in rfm_data.columns:
                                    customer_details['RFM_segment_label'] = rfm_data['RFM_segment_label']
                            else:
                                customer_details = pd.DataFrame()

                            if not customer_details.empty:
                                customer_details = customer_details.merge(customer_nights, on='Customer ID', how='right').fillna(0)
                            else:
                                customer_details = customer_nights

                            st.write(customer_details)
                            csv_data = convert_df(customer_details)
                            excel_data = convert_df_to_excel(customer_details)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(label="Download data as CSV", data=csv_data, file_name='seller_analysis.csv', mime='text/csv')
                            with col2:
                                st.download_button(label="Download data as Excel", data=excel_data, file_name='seller_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                            # Time Series
                            st.subheader("Time Series Analysis of Sales")
                            daily_df = st.session_state.single_seller_daily_df
                            if daily_df is None or daily_df.empty:
                                st.info("No time-series data available for this seller.")
                            else:
                                days_in_range = (end_date - start_date).days + 1
                                kpi_options = ['Total Deals','Successful Deals','New Customers','Returning Customers','Average Deal Value','Average Nights']
                                selected_kpis_to_plot = st.multiselect(
                                    "Select KPI(s) to Plot (no page reset):", 
                                    kpi_options, 
                                    default=['Total Deals','Successful Deals'], 
                                    key='single_seller_ts_kpis'
                                )
                                if selected_kpis_to_plot:
                                    for c in selected_kpis_to_plot:
                                        df_p = daily_df[['Date', c]].copy()
                                        df_p.sort_values('Date', inplace=True)
                                        if days_in_range < 60:
                                            df_p[c+'_7d_MA'] = df_p[c].rolling(7).mean()
                                            fig_ts = px.line(
                                                df_p,
                                                x='Date',
                                                y=[c, c+'_7d_MA'],
                                                title=f"Time Series of {c} (with 7 day MA)",
                                                labels={'value': f"{c}"},
                                                color_discrete_sequence=px.colors.qualitative.Set1
                                            )
                                        else:
                                            df_p[c+'_30d_MA'] = df_p[c].rolling(30).mean()
                                            fig_ts = px.line(
                                                df_p,
                                                x='Date',
                                                y=[c, c+'_30d_MA'],
                                                title=f"Time Series of {c} (with 30 day MA)",
                                                labels={'value': f"{c}"},
                                                color_discrete_sequence=px.colors.qualitative.Set1
                                            )
                                        st.plotly_chart(fig_ts)

                ###########################################################################
                #  COMPARE TWO SELLERS
                ###########################################################################
                with tabs[1]:
                    vip_options_compare_two = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_compare_two = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_seller_compare_two')
                    if select_all_vips_compare_two:
                        selected_vips_seller_compare_two = vip_options_compare_two
                    else:
                        selected_vips_seller_compare_two = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_compare_two,
                            default=[],
                            key='vips_multiselect_seller_compare_two'
                        )

                    s1, s2 = None, None  # define them up front to avoid scope errors

                    with st.form(key='compare_two_sellers_form', clear_on_submit=False):
                        two_sellers = st.multiselect("Select Two Sellers:", options=sellers_options, key='two_sellers_select', max_selections=2)
                        min_date_compare = data['تاریخ انجام معامله'].min()
                        max_date_compare = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date_compare) or pd.isna(max_date_compare):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date_compare = min_date_compare.date()
                        max_date_compare = max_date_compare.date()

                        start_date_compare = st.date_input("Start Date", value=min_date_compare, min_value=min_date_compare, max_value=max_date_compare, key='compare_two_start_date')
                        end_date_compare = st.date_input("End Date", value=max_date_compare, min_value=min_date_compare, max_value=max_date_compare, key='compare_two_end_date')

                        compare_kpi_options = [
                            'Total Deals','Successful Deals','Average Deal Value','Average Nights',
                            'Extension Rate','New Customers','Returning Customers'
                        ]
                        selected_compare_kpis = st.multiselect(
                            "Select KPI(s) to Plot",
                            compare_kpi_options,
                            default=['Average Deal Value','Extension Rate'],
                            key='compare_two_sellers_kpis'
                        )
                        apply_compare_two = st.form_submit_button(label='Compare')

                    if "two_sellers_results" not in st.session_state:
                        st.session_state.two_sellers_results = None
                        st.session_state.two_sellers_tsdata = None

                    if apply_compare_two:
                        if len(two_sellers) == 2:
                            if selected_vips_seller_compare_two:
                                s1 = two_sellers[0]
                                s2 = two_sellers[1]

                                df_s1 = data[
                                    (data['مسئول معامله'] == s1) &
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_compare)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_compare)) &
                                    (data['VIP Status'].isin(selected_vips_seller_compare_two))
                                ]
                                df_s2 = data[
                                    (data['مسئول معامله'] == s2) &
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_compare)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_compare)) &
                                    (data['VIP Status'].isin(selected_vips_seller_compare_two))
                                ]
                                df_s1_success = df_s1[df_s1['وضعیت معامله'] == 'موفق']
                                df_s2_success = df_s2[df_s2['وضعیت معامله'] == 'موفق']

                                def kpi_calc(df_all, df_succ):
                                    td = len(df_all)
                                    sd = len(df_succ)
                                    avgv = df_succ['ارزش معامله'].mean() if not df_succ.empty else 0
                                    nights_ = df_succ['تعداد شب'].mean() if not df_succ.empty else 0
                                    ext = df_succ[df_succ['نوع خرید'] == 'تمدید']
                                    ex_count = len(ext)
                                    ex_rate = (ex_count/sd)*100 if sd>0 else 0
                                    nw = 0
                                    rt = 0
                                    if not df_succ.empty:
                                        earliest_global = global_first_deal_date_series.to_dict()
                                        for cid_ in df_succ['کد دیدار شخص معامله'].unique():
                                            fd_ = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fd_):
                                                if start_date_compare <= fd_.date() <= end_date_compare:
                                                    nw += 1
                                                elif fd_.date() < start_date_compare:
                                                    rt += 1
                                    return {
                                        'Seller': (df_all['مسئول معامله'].iloc[0] if not df_all.empty else ""),
                                        'Total Deals': td,
                                        'Successful Deals': sd,
                                        'Average Deal Value': avgv,
                                        'Average Nights': nights_,
                                        'Extension Rate': ex_rate,
                                        'New Customers': nw,
                                        'Returning Customers': rt
                                    }

                                s1_stats = kpi_calc(df_s1, df_s1_success)
                                s2_stats = kpi_calc(df_s2, df_s2_success)

                                st.session_state.two_sellers_results = (s1_stats, s2_stats)

                                if not df_s1_success.empty:
                                    df_s1_success = df_s1_success.copy()
                                    df_s1_success['Seller'] = s1
                                if not df_s2_success.empty:
                                    df_s2_success = df_s2_success.copy()
                                    df_s2_success['Seller'] = s2
                                combined_success = pd.concat([df_s1_success, df_s2_success], ignore_index=True)
                                st.session_state.two_sellers_tsdata = combined_success
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select exactly two sellers.")

                    if st.session_state.two_sellers_results is not None:
                        s1_res, s2_res = st.session_state.two_sellers_results
                        if s1_res['Seller'] and s2_res['Seller']:
                            st.markdown("### Comparison KPIs")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**{s1_res['Seller']}**")
                                st.metric("Total Deals", f"{s1_res['Total Deals']}")
                                st.metric("Successful Deals", f"{s1_res['Successful Deals']}")
                                st.metric("Average Deal Value", f"{s1_res['Average Deal Value']:.0f}")
                                st.metric("Average Nights", f"{s1_res['Average Nights']:.2f}")
                                st.metric("Extension Rate (%)", f"{s1_res['Extension Rate']:.2f}%")
                                st.metric("New Customers", f"{s1_res['New Customers']}")
                                st.metric("Returning Customers", f"{s1_res['Returning Customers']}")
                            with c2:
                                st.markdown(f"**{s2_res['Seller']}**")
                                st.metric("Total Deals", f"{s2_res['Total Deals']}")
                                st.metric("Successful Deals", f"{s2_res['Successful Deals']}")
                                st.metric("Average Deal Value", f"{s2_res['Average Deal Value']:.0f}")
                                st.metric("Average Nights", f"{s2_res['Average Nights']:.2f}")
                                st.metric("Extension Rate (%)", f"{s2_res['Extension Rate']:.2f}%")
                                st.metric("New Customers", f"{s2_res['New Customers']}")
                                st.metric("Returning Customers", f"{s2_res['Returning Customers']}")

                            st.write("---")
                            st.markdown("**Direct Comparison of Each KPI**")

                            # We'll normalize each KPI in the bar chart so that the max in that KPI is 1.0
                            comp_data = [
                                {'KPI':'Total Deals','Seller':s1_res['Seller'],'Value': s1_res['Total Deals']},
                                {'KPI':'Total Deals','Seller':s2_res['Seller'],'Value': s2_res['Total Deals']},
                                {'KPI':'Successful Deals','Seller':s1_res['Seller'],'Value': s1_res['Successful Deals']},
                                {'KPI':'Successful Deals','Seller':s2_res['Seller'],'Value': s2_res['Successful Deals']},
                                {'KPI':'Average Deal Value','Seller':s1_res['Seller'],'Value': s1_res['Average Deal Value']},
                                {'KPI':'Average Deal Value','Seller':s2_res['Seller'],'Value': s2_res['Average Deal Value']},
                                {'KPI':'Average Nights','Seller':s1_res['Seller'],'Value': s1_res['Average Nights']},
                                {'KPI':'Average Nights','Seller':s2_res['Seller'],'Value': s2_res['Average Nights']},
                                {'KPI':'Extension Rate','Seller':s1_res['Seller'],'Value': s1_res['Extension Rate']},
                                {'KPI':'Extension Rate','Seller':s2_res['Seller'],'Value': s2_res['Extension Rate']},
                                {'KPI':'New Customers','Seller':s1_res['Seller'],'Value': s1_res['New Customers']},
                                {'KPI':'New Customers','Seller':s2_res['Seller'],'Value': s2_res['New Customers']},
                                {'KPI':'Returning Customers','Seller':s1_res['Seller'],'Value': s1_res['Returning Customers']},
                                {'KPI':'Returning Customers','Seller':s2_res['Seller'],'Value': s2_res['Returning Customers']},
                            ]
                            comp_df_side = pd.DataFrame(comp_data)

                            # For each KPI, find the max, then create a new column "Normalized Value"
                            comp_df_list = []
                            for kpi_name in comp_df_side['KPI'].unique():
                                sub = comp_df_side[comp_df_side['KPI'] == kpi_name].copy()
                                max_val = sub['Value'].max()
                                if max_val == 0:
                                    sub['Normalized Value'] = 0
                                else:
                                    sub['Normalized Value'] = sub['Value']/max_val
                                comp_df_list.append(sub)
                            comp_df_side_final = pd.concat(comp_df_list, ignore_index=True)

                            fig_kpi_compare = px.bar(
                                comp_df_side_final,
                                x='KPI',
                                y='Normalized Value',
                                color='Seller',
                                barmode='group',
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                title="Side-by-Side KPI Comparison (Normalized)"
                            )
                            fig_kpi_compare.update_traces(
                                hovertemplate='<b>KPI</b>: %{x}<br><b>Seller</b>: %{color}<br>Value: %{customdata[0]}<extra></extra>',
                                customdata=np.expand_dims(comp_df_side_final['Value'], axis=1)
                            )
                            st.plotly_chart(fig_kpi_compare)

                    # Show Time-Series if we have them
                    if (
                        'two_sellers_tsdata' in st.session_state and
                        st.session_state.two_sellers_tsdata is not None and
                        not st.session_state.two_sellers_tsdata.empty and
                        selected_compare_kpis
                    ):
                        ts_df = st.session_state.two_sellers_tsdata.copy()
                        ts_df['Date'] = pd.to_datetime(ts_df['تاریخ انجام معامله'], errors='coerce')
                        ts_df.dropna(subset=['Date'], inplace=True)

                        st.markdown("### Time Series Comparison (Each KPI in its own plot)")

                        days_in_range_compare = (end_date_compare - start_date_compare).days + 1
                        # We'll color the raw lines in a bold color, the moving average in a pastel variant
                        # We'll define fallback color pairs in case we only have 1 or 2 sellers
                        # We'll detect the sellers in the actual df
                        existing_sellers_in_ts = ts_df['Seller'].unique()

                        # We'll define a function that given a seller name, returns a (raw, pastel) color
                        def get_seller_colors(sname):
                            # fallback color pairs
                            color_pairs = {
                                s1_res['Seller'] if s1_res else 'SellerA': ('#d62728','#ffa09e'),  # bold red, pastel red
                                s2_res['Seller'] if s2_res else 'SellerB': ('#1f77b4','#aec7e8'),  # bold blue, pastel
                            }
                            # fallback if not found
                            return color_pairs.get(sname, ('#2ca02c','#98df8a'))

                        for k in selected_compare_kpis:
                            # We'll produce daily-level stats for each day, for each Seller, for that KPI
                            # Then only show 30-day MA if date range >= 60, else show 7-day

                            day_list = pd.date_range(start=start_date_compare, end=end_date_compare, freq='D')
                            daily_list = []
                            earliest_global = global_first_deal_date_series.to_dict()
                            sub_columns = ts_df[['Date','Seller','کد دیدار شخص معامله','ارزش معامله','نوع خرید','تعداد شب']].copy()

                            for dday in day_list:
                                day_sub = sub_columns[sub_columns['Date'].dt.date == dday.date()]
                                # We'll separate by each seller in day_sub
                                for seller_ in day_sub['Seller'].unique():
                                    sub2 = day_sub[day_sub['Seller'] == seller_]
                                    val = 0
                                    if k == 'Average Deal Value':
                                        val = sub2['ارزش معامله'].sum()/len(sub2) if len(sub2)>0 else 0
                                    elif k == 'Extension Rate':
                                        ex_cnt = len(sub2[sub2['نوع خرید']=='تمدید'])
                                        tot_cnt = len(sub2)
                                        val = (ex_cnt/tot_cnt*100) if tot_cnt>0 else 0
                                    elif k == 'Average Nights':
                                        val = sub2['تعداد شب'].mean() if len(sub2)>0 else 0
                                    elif k == 'Total Deals':
                                        val = len(sub2)
                                    elif k == 'Successful Deals':
                                        val = len(sub2)
                                    elif k == 'New Customers':
                                        newC = 0
                                        for cid_ in sub2['کد دیدار شخص معامله'].unique():
                                            fdate = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fdate) and fdate.date() == dday.date():
                                                newC += 1
                                        val = newC
                                    elif k == 'Returning Customers':
                                        retC = 0
                                        for cid_ in sub2['کد دیدار شخص معامله'].unique():
                                            fdate = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fdate) and fdate.date() < dday.date():
                                                retC += 1
                                        val = retC

                                    daily_list.append({
                                        'Date': dday,
                                        'Seller': seller_,
                                        'Value': val
                                    })
                                # Also account for if a seller is missing on that day => value=0
                                # We'll cross-check existing sellers vs. day_sub
                                for sell_ in existing_sellers_in_ts:
                                    # If that seller doesn't appear in day_sub
                                    if sell_ not in day_sub['Seller'].unique():
                                        daily_list.append({'Date': dday, 'Seller': sell_, 'Value': 0})

                            daily_k_df = pd.DataFrame(daily_list)
                            daily_k_df.sort_values(['Seller','Date'], inplace=True)

                            if days_in_range_compare < 60:
                                daily_k_df['MA'] = daily_k_df.groupby('Seller')['Value'].transform(lambda x: x.rolling(7).mean())
                                nameMA = '7d MA'
                            else:
                                daily_k_df['MA'] = daily_k_df.groupby('Seller')['Value'].transform(lambda x: x.rolling(30).mean())
                                nameMA = '30d MA'

                            # We'll build a custom figure
                            fig_ts = go.Figure()
                            fig_ts.update_layout(
                                title=f"{k} Over Time",
                                xaxis_title="Date",
                                yaxis_title=f"{k}"
                            )

                            # We'll get unique sellers
                            for seller_ in daily_k_df['Seller'].unique():
                                sub_seller = daily_k_df[daily_k_df['Seller'] == seller_]
                                raw_color, pastel_color = get_seller_colors(seller_)

                                # raw line
                                fig_ts.add_trace(go.Scatter(
                                    x=sub_seller['Date'],
                                    y=sub_seller['Value'],
                                    mode='lines+markers',
                                    name=f"{seller_} - raw {k}",
                                    line=dict(color=raw_color, width=2),
                                    marker=dict(color=raw_color, size=5)
                                ))
                                # MA line
                                fig_ts.add_trace(go.Scatter(
                                    x=sub_seller['Date'],
                                    y=sub_seller['MA'],
                                    mode='lines',
                                    name=f"{seller_} - {nameMA}",
                                    line=dict(color=pastel_color, width=3, dash='dot')
                                ))

                            st.plotly_chart(fig_ts)

                ###########################################################################
                #  COMPARE ALL SELLERS
                ###########################################################################
                with tabs[2]:
                    vip_options_compare_all = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_compare_all = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_seller_compare_all')
                    if select_all_vips_compare_all:
                        selected_vips_seller_compare_all = vip_options_compare_all
                    else:
                        selected_vips_seller_compare_all = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_compare_all,
                            default=[],
                            key='vips_multiselect_seller_compare_all'
                        )

                    with st.form(key='compare_all_sellers_form', clear_on_submit=False):
                        min_date_all = data['تاریخ انجام معامله'].min()
                        max_date_all = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date_all) or pd.isna(max_date_all):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date_all = min_date_all.date()
                        max_date_all = max_date_all.date()

                        start_date_all = st.date_input("Start Date", value=min_date_all, min_value=min_date_all, max_value=max_date_all, key='compare_all_start_date')
                        end_date_all = st.date_input("End Date", value=max_date_all, min_value=min_date_all, max_value=max_date_all, key='compare_all_end_date')
                        apply_compare_all = st.form_submit_button(label='Compare All Sellers')

                    if "compare_all_results" not in st.session_state:
                        st.session_state.compare_all_results = None

                    if apply_compare_all:
                        if selected_vips_seller_compare_all:
                            all_sellers_data = data[
                                (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_all)) &
                                (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_all)) &
                                (data['VIP Status'].isin(selected_vips_seller_compare_all))
                            ]
                            if all_sellers_data.empty:
                                st.warning("No deals found for the selected VIP statuses in the specified date range.")
                            else:
                                sellers_list = all_sellers_data['مسئول معامله'].unique().tolist()
                                final_rows = []
                                for sel in sellers_list:
                                    sel_df = all_sellers_data[all_sellers_data['مسئول معامله'] == sel]
                                    sel_suc = sel_df[sel_df['وضعیت معامله'] == 'موفق']
                                    td_ = len(sel_df)
                                    sd_ = len(sel_suc)
                                    sr_ = (sd_/td_)*100 if td_>0 else 0
                                    av_ = sel_suc['ارزش معامله'].mean() if not sel_suc.empty else 0
                                    ni_ = sel_suc['تعداد شب'].mean() if not sel_suc.empty else 0
                                    ex_ = sel_suc[sel_suc['نوع خرید'] == 'تمدید']
                                    ex_cnt_ = len(ex_)
                                    ex_rate_ = (ex_cnt_/sd_)*100 if sd_>0 else 0
                                    n_c = 0
                                    r_c = 0
                                    if not sel_suc.empty:
                                        for cc in sel_suc['کد دیدار شخص معامله'].unique():
                                            fd = global_first_deal_date_series.get(cc, pd.NaT)
                                            if not pd.isna(fd):
                                                if start_date_all <= fd.date() <= end_date_all:
                                                    n_c += 1
                                                elif fd.date() < start_date_all:
                                                    r_c += 1
                                    final_rows.append({
                                        'Seller': sel,
                                        'Total Deals': td_,
                                        'Successful Deals': sd_,
                                        'Success Rate': sr_,
                                        'Avg Deal Value': av_,
                                        'Avg Nights': ni_,
                                        'Extension Rate': ex_rate_,
                                        'New Customers': n_c,
                                        'Returning Customers': r_c
                                    })
                                comp_df = pd.DataFrame(final_rows)
                                st.session_state.compare_all_results = comp_df
                        else:
                            st.warning("Please select at least one VIP status.")

                    if st.session_state.compare_all_results is not None and not st.session_state.compare_all_results.empty:
                        comp_df = st.session_state.compare_all_results
                        st.write("### All Sellers Comparison")
                        st.write(comp_df)

                        c_csv = convert_df(comp_df)
                        c_excel = convert_df_to_excel(comp_df)
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.download_button(label="Download as CSV", data=c_csv, file_name='all_sellers_comparison.csv', mime='text/csv')
                        with cc2:
                            st.download_button(label="Download as Excel", data=c_excel, file_name='all_sellers_comparison.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                        if not comp_df.empty:
                            fig_all_sellers = px.bar(
                                comp_df,
                                x='Seller',
                                y='Successful Deals',
                                title="Successful Deals by Seller",
                                text='Successful Deals',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_sellers.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_sellers)

                            fig_all_sellers_val = px.bar(
                                comp_df,
                                x='Seller',
                                y='Avg Deal Value',
                                title="Average Deal Value by Seller",
                                text='Avg Deal Value',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_sellers_val.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_sellers_val)

                            fig_all_sellers_sr = px.bar(
                                comp_df,
                                x='Seller',
                                y='Success Rate',
                                title="Success Rate (%) by Seller",
                                text='Success Rate',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_sellers_sr.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_sellers_sr)

                            fig_all_ext = px.bar(
                                comp_df,
                                x='Seller',
                                y='Extension Rate',
                                title="Extension Rate (%) by Seller",
                                text='Extension Rate',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_ext.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_ext)

                            # Show separate box plots for each metric
                            st.markdown("#### Separate Box Plots for Key Metrics")
                            for metric in ['Avg Deal Value','Avg Nights','Success Rate','Extension Rate']:
                                bx_df = comp_df[['Seller', metric]].copy()
                                fig_box = px.box(
                                    bx_df,
                                    x='Seller',
                                    y=metric,
                                    color='Seller',
                                    color_discrete_sequence=px.colors.qualitative.Set1,
                                    title=f"Box Plot of {metric} by Seller"
                                )
                                st.plotly_chart(fig_box)

                ###########################################################################
                #  RFM SALES ANALYSIS
                ###########################################################################
                with tabs[3]:
                    st.subheader("RFM Sales Analysis")

                    select_all_clusters_seller = st.checkbox("Select all clusters", value=True, key='select_all_clusters_seller')
                    if select_all_clusters_seller:
                        if 'RFM_segment_label' in rfm_data.columns:
                            selected_clusters_seller = sorted(rfm_data['RFM_segment_label'].unique().tolist())
                        else:
                            selected_clusters_seller = []
                    else:
                        if 'RFM_segment_label' in rfm_data.columns:
                            all_segments = sorted(rfm_data['RFM_segment_label'].unique().tolist())
                        else:
                            all_segments = []
                        selected_clusters_seller = st.multiselect(
                            "Select Clusters:",
                            options=all_segments,
                            default=[],
                            key='clusters_multiselect_seller'
                        )

                    vip_options_page_cluster = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_page_cluster = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_seller_cluster')
                    if select_all_vips_page_cluster:
                        selected_vips_seller_cluster = vip_options_page_cluster
                    else:
                        selected_vips_seller_cluster = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_page_cluster,
                            default=[],
                            key='vips_multiselect_seller_cluster'
                        )

                    with st.form(key='seller_cluster_form', clear_on_submit=False):
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='seller_cluster_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='seller_cluster_end_date')

                        apply_cluster_filters = st.form_submit_button(label='Apply Filters')

                    if "rfm_sales_data" not in st.session_state:
                        st.session_state.rfm_sales_data = None
                        st.session_state.rfm_sales_kpis = None

                    if apply_cluster_filters:
                        if len(selected_clusters_seller) == 0:
                            st.warning("Please select at least one cluster (or ensure RFM_segment_label is present in rfm_data).")
                            st.session_state.rfm_sales_data = None
                            st.session_state.rfm_sales_kpis = None
                        else:
                            if selected_vips_seller_cluster:
                                date_filtered_data_all = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date))
                                ]
                                if 'RFM_segment_label' not in rfm_data.columns:
                                    st.error("RFM_segment_label column not found in rfm_data. Cannot filter by cluster.")
                                    st.session_state.rfm_sales_data = None
                                    st.session_state.rfm_sales_kpis = None
                                else:
                                    cluster_customers = rfm_data[rfm_data['RFM_segment_label'].isin(selected_clusters_seller)]['Customer ID'].unique()
                                    cluster_deals_all = date_filtered_data_all[
                                        date_filtered_data_all['کد دیدار شخص معامله'].isin(cluster_customers) &
                                        date_filtered_data_all['VIP Status'].isin(selected_vips_seller_cluster)
                                    ]
                                    if cluster_deals_all.empty:
                                        st.warning("No deals found for the selected clusters and VIP statuses in the specified date range.")
                                        st.session_state.rfm_sales_data = None
                                        st.session_state.rfm_sales_kpis = None
                                    else:
                                        cluster_deals = cluster_deals_all[cluster_deals_all['وضعیت معامله'] == 'موفق']
                                        total_deals = len(cluster_deals_all)
                                        successful_deals_count = len(cluster_deals)
                                        success_rate = (successful_deals_count / total_deals)*100 if total_deals>0 else 0
                                        new_customers = 0
                                        returning_customers = 0
                                        if not cluster_deals.empty:
                                            unique_customers = cluster_deals['کد دیدار شخص معامله'].unique()
                                            for cid in unique_customers:
                                                first_deal_date = global_first_deal_date_series.get(cid, pd.NaT)
                                                if pd.isna(first_deal_date):
                                                    continue
                                                if start_date <= first_deal_date.date() <= end_date:
                                                    new_customers += 1
                                                elif first_deal_date.date() < start_date:
                                                    returning_customers += 1

                                        avg_deal_value = cluster_deals['ارزش معامله'].mean() if not cluster_deals.empty else 0
                                        avg_nights = cluster_deals['تعداد شب'].mean() if not cluster_deals.empty else 0
                                        cluster_extentions = cluster_deals[cluster_deals['نوع خرید'] == 'تمدید']
                                        cluster_extentions_count = len(cluster_extentions)
                                        cluster_extention_rate = (cluster_extentions_count / successful_deals_count*100) if successful_deals_count>0 else 0

                                        prev_length = (end_date - start_date).days + 1
                                        prev_end = start_date - timedelta(days=1)
                                        prev_start = prev_end - timedelta(days=prev_length - 1)
                                        prev_data_all = data[
                                            (data['تاریخ انجام معامله'] >= pd.to_datetime(prev_start)) &
                                            (data['تاریخ انجام معامله'] <= pd.to_datetime(prev_end))
                                        ]
                                        prev_data_all = prev_data_all[
                                            prev_data_all['کد دیدار شخص معامله'].isin(cluster_customers) &
                                            prev_data_all['VIP Status'].isin(selected_vips_seller_cluster)
                                        ]
                                        prev_deals = prev_data_all[prev_data_all['وضعیت معامله'] == 'موفق']
                                        if not prev_data_all.empty:
                                            ptd = len(prev_data_all)
                                            psd = len(prev_deals)
                                            psr = (psd / ptd)*100 if ptd>0 else 0
                                            pav = prev_deals['ارزش معامله'].mean() if not prev_deals.empty else 0
                                            pni = prev_deals['تعداد شب'].mean() if not prev_deals.empty else 0
                                            pext = prev_deals[prev_deals['نوع خرید'] == 'تمدید']
                                            pext_cnt = len(pext)
                                            pext_rate = (pext_cnt/psd*100) if psd>0 else 0
                                            pnew_c = 0
                                            pret_c = 0
                                            if not prev_deals.empty:
                                                for p_cid in prev_deals['کد دیدار شخص معامله'].unique():
                                                    fd = global_first_deal_date_series.get(p_cid, pd.NaT)
                                                    if not pd.isna(fd):
                                                        if prev_start <= fd.date() <= prev_end:
                                                            pnew_c += 1
                                                        elif fd.date() < prev_start:
                                                            pret_c += 1
                                        else:
                                            ptd = 0
                                            psd = 0
                                            psr = 0
                                            pav = 0
                                            pni = 0
                                            pext_rate = 0
                                            pnew_c = 0
                                            pret_c = 0

                                        st.session_state.rfm_sales_data = cluster_deals.copy()
                                        st.session_state.rfm_sales_kpis = {
                                            'total_deals': total_deals,
                                            'successful_deals_count': successful_deals_count,
                                            'success_rate': success_rate,
                                            'avg_deal_value': avg_deal_value,
                                            'avg_nights': avg_nights,
                                            'cluster_extention_rate': cluster_extention_rate,
                                            'new_customers': new_customers,
                                            'returning_customers': returning_customers,
                                            'ptd': ptd,
                                            'psd': psd,
                                            'psr': psr,
                                            'pav': pav,
                                            'pni': pni,
                                            'pext_rate': pext_rate,
                                            'pnew_c': pnew_c,
                                            'pret_c': pret_c
                                        }
                            else:
                                st.warning("Please select at least one VIP status.")

                    if st.session_state.rfm_sales_data is not None and st.session_state.rfm_sales_kpis is not None:
                        cluster_deals = st.session_state.rfm_sales_data
                        kpis = st.session_state.rfm_sales_kpis

                        total_deals = kpis['total_deals']
                        successful_deals_count = kpis['successful_deals_count']
                        success_rate = kpis['success_rate']
                        avg_deal_value = kpis['avg_deal_value']
                        avg_nights = kpis['avg_nights']
                        cluster_extention_rate = kpis['cluster_extention_rate']
                        new_customers = kpis['new_customers']
                        returning_customers = kpis['returning_customers']
                        ptd = kpis['ptd']
                        psd = kpis['psd']
                        psr = kpis['psr']
                        pav = kpis['pav']
                        pni = kpis['pni']
                        pext_rate = kpis['pext_rate']
                        pnew_c = kpis['pnew_c']
                        pret_c = kpis['pret_c']

                        def pdiff(x, y):
                            if y == 0:
                                return None
                            return f"{((x-y)/abs(y)*100):.2f}%"

                        colKPI1, colKPI2, colKPI3, colKPI4 = st.columns(4)
                        colKPI1.metric(
                            "Total Deals", 
                            f"{total_deals}", 
                            pdiff(total_deals, ptd)
                        )
                        colKPI2.metric(
                            "Successful Deals", 
                            f"{successful_deals_count}",
                            pdiff(successful_deals_count, psd)
                        )
                        colKPI3.metric(
                            "Success Rate (%)",
                            f"{success_rate:.2f}%",
                            pdiff(success_rate, psr)
                        )
                        colKPI4.metric(
                            "Avg. Deal Value",
                            f"{avg_deal_value:,.0f}",
                            pdiff(avg_deal_value, pav)
                        )

                        colKPI5, colKPI6, colKPI7, colKPI8 = st.columns(4)
                        colKPI5.metric(
                            "New Customers",
                            f"{new_customers}",
                            pdiff(new_customers, pnew_c)
                        )
                        colKPI6.metric(
                            "Returning Customers",
                            f"{returning_customers}",
                            pdiff(returning_customers, pret_c)
                        )
                        colKPI7.metric(
                            "Avg. Nights",
                            f"{avg_nights:.2f}",
                            pdiff(avg_nights, pni)
                        )
                        colKPI8.metric(
                            "Extention Rate",
                            f"{cluster_extention_rate:.2f}%",
                            pdiff(cluster_extention_rate, pext_rate)
                        )

                        st.write("---")
                        if cluster_deals.empty:
                            st.warning("No successful deals found for these clusters in the specified date range.")
                        else:
                            seller_counts = cluster_deals['مسئول معامله'].value_counts().reset_index()
                            seller_counts.columns = ['Seller','Count']
                            fig_seller_cluster_freq = px.bar(
                                seller_counts,
                                x='Seller',
                                y='Count',
                                title="Seller Distribution (Frequency)",
                                labels={'Seller': 'Seller','Count': 'Number of Deals'},
                                text='Count',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_seller_cluster_freq.update_traces(textposition='outside')
                            st.plotly_chart(fig_seller_cluster_freq)

                            seller_monetary = cluster_deals.groupby('مسئول معامله')['ارزش معامله'].sum().reset_index()
                            seller_monetary.columns = ['Seller','Monetary']
                            fig_seller_cluster_monetary = px.bar(
                                seller_monetary,
                                x='Seller',
                                y='Monetary',
                                title="Seller Distribution (Monetary)",
                                labels={'Seller': 'Seller','Monetary': 'Total Monetary Value'},
                                text='Monetary',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_seller_cluster_monetary.update_traces(textposition='outside')
                            st.plotly_chart(fig_seller_cluster_monetary)

                            st.subheader("Successful Deals")
                            if 'RFM_segment_label' in rfm_data.columns:
                                cluster_deals = cluster_deals.merge(
                                    rfm_data[['Customer ID','RFM_segment_label']],
                                    left_on='کد دیدار شخص معامله',
                                    right_on='Customer ID',
                                    how='left'
                                )
                            if 'RFM_segment_label' in cluster_deals.columns:
                                deals_table = cluster_deals[[
                                    'Customer ID','نام شخص معامله','نام خانوادگی شخص معامله',
                                    'موبایل شخص معامله','VIP Status','RFM_segment_label',
                                    'مسئول معامله','تعداد شب','ارزش معامله','تاریخ انجام معامله'
                                ]]
                            else:
                                deals_table = cluster_deals

                            st.write(deals_table)
                            csv_data = convert_df(deals_table)
                            excel_data = convert_df_to_excel(deals_table)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(label="Download data as CSV", data=csv_data, file_name='seller_cluster_deals.csv', mime='text/csv')
                            with col2:
                                st.download_button(label="Download data as Excel", data=excel_data, file_name='seller_cluster_deals.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                            st.subheader("Time Series Analysis of Sales")
                            cluster_deals_time_df = cluster_deals[['تاریخ انجام معامله','ارزش معامله']].copy()
                            cluster_deals_time_df['تاریخ انجام معامله'] = pd.to_datetime(cluster_deals_time_df['تاریخ انجام معامله'], errors='coerce')
                            cluster_deals_time_df.dropna(subset=['تاریخ انجام معامله'], inplace=True)
                            if cluster_deals_time_df.empty:
                                st.info("No time-series data available for these clusters.")
                            else:
                                cluster_deals_time_df = cluster_deals_time_df.groupby(cluster_deals_time_df['تاریخ انجام معامله'].dt.date)['ارزش معامله'].sum().reset_index()
                                cluster_deals_time_df.rename(columns={'تاریخ انجام معامله': 'Date','ارزش معامله': 'Sales'}, inplace=True)
                                cluster_deals_time_df['Date'] = pd.to_datetime(cluster_deals_time_df['Date'])
                                cluster_deals_time_df.sort_values('Date', inplace=True)

                                days_in_rfm = (end_date - start_date).days + 1
                                if days_in_rfm < 60:
                                    cluster_deals_time_df['7d_MA'] = cluster_deals_time_df['Sales'].rolling(7).mean()
                                    lines_to_use = ['Sales','7d_MA']
                                    chart_title = "Daily Sales Over Time (with 7-day MA)"
                                else:
                                    cluster_deals_time_df['30d_MA'] = cluster_deals_time_df['Sales'].rolling(30).mean()
                                    lines_to_use = ['Sales','30d_MA']
                                    chart_title = "Daily Sales Over Time (with 30-day MA)"

                                fig_cluster_time = px.line(
                                    cluster_deals_time_df,
                                    x='Date',
                                    y=lines_to_use,
                                    labels={'value': 'Sales Amount'},
                                    title=chart_title,
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                st.plotly_chart(fig_cluster_time)

                                monthly_df = cluster_deals_time_df[['Date','Sales']].copy()
                                monthly_df['Month'] = monthly_df['Date'].dt.to_period('M')
                                monthly_avg = monthly_df.groupby('Month')['Sales'].mean().reset_index()
                                monthly_avg['Month'] = monthly_avg['Month'].astype(str)
                                fig_cluster_monthly = px.bar(
                                    monthly_avg,
                                    x='Month',
                                    y='Sales',
                                    labels={'Sales': 'Average Sales'},
                                    title="Monthly Average Sales",
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                st.plotly_chart(fig_cluster_monthly)

                                total_sales_time = cluster_deals_time_df['Sales'].sum()
                                avg_sales_time = cluster_deals_time_df['Sales'].mean()

                                colA, colB = st.columns(2)
                                with colA:
                                    st.metric("Total Sales (Selected Period)", f"{total_sales_time:,.0f}")
                                with colB:
                                    st.metric("Avg Daily Sales (Selected Period)", f"{avg_sales_time:,.2f}")

                        ##########################################
            #          Sale Channel Analysis         #
            ##########################################

            elif page == 'Sale Channel Analysis':
                st.subheader("Sale Channel Analysis")

                # We use tabs for the four sections
                tabs = st.tabs([
                    "Single Channel Analysis",
                    "Compare Two Channels",
                    "Compare All Channels",
                    "RFM Sales Analysis",
                    "Channel Transitions"  # <-- new tab added here
                ])


                @st.cache_data
                def get_first_successful_deal_date_for_channels(df):
                    """Return a series mapping each customer to their first successful deal date."""
                    successful_deals_only = df[df['وضعیت معامله'] == 'موفق'].copy()
                    first_deal = successful_deals_only.groupby('کد دیدار شخص معامله')['تاریخ انجام معامله'].min()
                    return first_deal

                global_first_deal_date_series_channels = get_first_successful_deal_date_for_channels(data)

                ###########################################################################
                #  SINGLE CHANNEL ANALYSIS
                ###########################################################################
                with tabs[0]:
                    st.markdown("### Single Channel Analysis")

                    # VIP Filter
                    vip_options_page = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_page = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_channel_single')
                    if select_all_vips_page:
                        selected_vips_channel = vip_options_page
                    else:
                        selected_vips_channel = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_page,
                            default=[],
                            key='vips_multiselect_channel_single'
                        )

                    with st.form(key='channel_filters_form', clear_on_submit=False):
                        selected_channel = st.selectbox("Select a Sale Channel:", options=sale_channels_options)
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input(
                            "Start Date", 
                            value=min_date,
                            min_value=min_date, 
                            max_value=max_date, 
                            key='channel_start_date_single'
                        )
                        end_date = st.date_input(
                            "End Date", 
                            value=max_date,
                            min_value=min_date, 
                            max_value=max_date, 
                            key='channel_end_date_single'
                        )

                        apply_channel_filters = st.form_submit_button(label='Apply Filters')

                    if "single_channel_data" not in st.session_state:
                        st.session_state.single_channel_data = None
                        st.session_state.single_channel_filtered_all = None
                        st.session_state.single_channel_kpi_df = None
                        st.session_state.single_channel_daily_df = None

                    if apply_channel_filters:
                        if selected_channel:
                            if selected_vips_channel:
                                # Filter data
                                date_filtered_data_all = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date)) &
                                    (data['شیوه آشنایی معامله'] == selected_channel)
                                ]
                                date_filtered_data_all = date_filtered_data_all[date_filtered_data_all['VIP Status'].isin(selected_vips_channel)]
                                channel_data_success = date_filtered_data_all[date_filtered_data_all['وضعیت معامله'] == 'موفق']

                                if date_filtered_data_all.empty:
                                    st.warning("No deals found for this channel in the specified date range.")
                                    st.session_state.single_channel_data = None
                                    st.session_state.single_channel_filtered_all = None
                                    st.session_state.single_channel_kpi_df = None
                                    st.session_state.single_channel_daily_df = None
                                else:
                                    st.session_state.single_channel_filtered_all = date_filtered_data_all.copy()
                                    st.session_state.single_channel_data = channel_data_success.copy()

                                    # KPIs
                                    total_deals = len(date_filtered_data_all)
                                    successful_deals_count = len(channel_data_success)
                                    success_rate = (successful_deals_count / total_deals)*100 if total_deals>0 else 0

                                    new_customers = 0
                                    returning_customers = 0
                                    if not channel_data_success.empty:
                                        unique_customers = channel_data_success['کد دیدار شخص معامله'].unique()
                                        for cid in unique_customers:
                                            first_deal_date = global_first_deal_date_series_channels.get(cid, pd.NaT)
                                            if pd.isna(first_deal_date):
                                                continue
                                            if start_date <= first_deal_date.date() <= end_date:
                                                new_customers += 1
                                            elif first_deal_date.date() < start_date:
                                                returning_customers += 1

                                    avg_deal_value = channel_data_success['ارزش معامله'].mean() if not channel_data_success.empty else 0
                                    avg_nights = channel_data_success['تعداد شب'].mean() if not channel_data_success.empty else 0

                                    # Extension analysis
                                    ext = channel_data_success[channel_data_success['نوع خرید'] == 'تمدید']
                                    ext_cnt = len(ext)
                                    ext_rate = (ext_cnt/successful_deals_count*100) if successful_deals_count>0 else 0

                                    # Compare with previous period
                                    prev_period_length = (end_date - start_date).days + 1
                                    prev_end_date = start_date - timedelta(days=1)
                                    prev_start_date = prev_end_date - timedelta(days=prev_period_length - 1)
                                    prev_period_data_all = data[
                                        (data['تاریخ انجام معامله'] >= pd.to_datetime(prev_start_date)) &
                                        (data['تاریخ انجام معامله'] <= pd.to_datetime(prev_end_date)) &
                                        (data['شیوه آشنایی معامله'] == selected_channel)
                                    ]
                                    prev_period_data_all = prev_period_data_all[prev_period_data_all['VIP Status'].isin(selected_vips_channel)]
                                    prev_channel_success = prev_period_data_all[prev_period_data_all['وضعیت معامله'] == 'موفق']

                                    if not prev_period_data_all.empty:
                                        prev_total_deals = len(prev_period_data_all)
                                        prev_successful_deals_count = len(prev_channel_success)
                                        prev_success_rate = (prev_successful_deals_count / prev_total_deals)*100 if prev_total_deals>0 else 0
                                        prev_avg_deal_value = prev_channel_success['ارزش معامله'].mean() if not prev_channel_success.empty else 0
                                        prev_avg_nights = prev_channel_success['تعداد شب'].mean() if not prev_channel_success.empty else 0
                                        prev_ext_ = prev_channel_success[prev_channel_success['نوع خرید'] == 'تمدید']
                                        prev_ext_cnt_ = len(prev_ext_)
                                        prev_ext_rate = (prev_ext_cnt_/prev_successful_deals_count*100) if prev_successful_deals_count>0 else 0

                                        prev_new_customers = 0
                                        prev_returning_customers = 0
                                        if not prev_channel_success.empty:
                                            unique_customers_prev = prev_channel_success['کد دیدار شخص معامله'].unique()
                                            for cid in unique_customers_prev:
                                                first_deal_date = global_first_deal_date_series_channels.get(cid, pd.NaT)
                                                if pd.isna(first_deal_date):
                                                    continue
                                                if prev_start_date <= first_deal_date.date() <= prev_end_date:
                                                    prev_new_customers += 1
                                                elif first_deal_date.date() < prev_start_date:
                                                    prev_returning_customers += 1
                                    else:
                                        prev_total_deals = 0
                                        prev_successful_deals_count = 0
                                        prev_success_rate = 0
                                        prev_avg_deal_value = 0
                                        prev_avg_nights = 0
                                        prev_ext_rate = 0
                                        prev_new_customers = 0
                                        prev_returning_customers = 0

                                    st.session_state.single_channel_kpi_df = {
                                        'total_deals': total_deals,
                                        'successful_deals_count': successful_deals_count,
                                        'success_rate': success_rate,
                                        'avg_deal_value': avg_deal_value,
                                        'avg_nights': avg_nights,
                                        'extention_rate': ext_rate,
                                        'new_customers': new_customers,
                                        'returning_customers': returning_customers,
                                        'prev_total_deals': prev_total_deals,
                                        'prev_successful_deals_count': prev_successful_deals_count,
                                        'prev_success_rate': prev_success_rate,
                                        'prev_avg_deal_value': prev_avg_deal_value,
                                        'prev_avg_nights': prev_avg_nights,
                                        'prev_extention_rate': prev_ext_rate,
                                        'prev_new_customers': prev_new_customers,
                                        'prev_returning_customers': prev_returning_customers
                                    }

                                    # Build daily metrics
                                    daily_metrics = []
                                    days_range = pd.date_range(start=start_date, end=end_date, freq='D')
                                    earliest_global = global_first_deal_date_series_channels.to_dict()
                                    for single_day in days_range:
                                        day_data_all = date_filtered_data_all[date_filtered_data_all['تاریخ انجام معامله'].dt.date == single_day.date()]
                                        day_data_success = day_data_all[day_data_all['وضعیت معامله'] == 'موفق']
                                        td = len(day_data_all)
                                        sd = len(day_data_success)
                                        dv = day_data_success['ارزش معامله'].mean() if not day_data_success.empty else 0
                                        nights_v = day_data_success['تعداد شب'].mean() if not day_data_success.empty else 0

                                        new_cus = 0
                                        ret_cus = 0
                                        if not day_data_success.empty:
                                            for ccid in day_data_success['کد دیدار شخص معامله'].unique():
                                                fdate = earliest_global.get(ccid, pd.NaT)
                                                if not pd.isna(fdate):
                                                    if single_day.date() == fdate.date():
                                                        new_cus += 1
                                                    elif fdate.date() < single_day.date():
                                                        ret_cus += 1

                                        daily_metrics.append({
                                            'Date': single_day,
                                            'Total Deals': td,
                                            'Successful Deals': sd,
                                            'New Customers': new_cus,
                                            'Returning Customers': ret_cus,
                                            'Average Deal Value': dv,
                                            'Average Nights': nights_v
                                        })
                                    daily_df = pd.DataFrame(daily_metrics)
                                    st.session_state.single_channel_daily_df = daily_df.copy()
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select a sale channel.")

                    # Display results
                    if (
                        st.session_state.single_channel_data is not None and
                        st.session_state.single_channel_filtered_all is not None and
                        st.session_state.single_channel_kpi_df is not None
                    ):
                        channel_data_success = st.session_state.single_channel_data
                        data_filtered_all = st.session_state.single_channel_filtered_all
                        kpi_data = st.session_state.single_channel_kpi_df

                        def pct_diff(new_val, old_val):
                            if old_val == 0:
                                return None
                            return f"{((new_val - old_val)/abs(old_val)*100):.2f}%"

                        total_deals = kpi_data['total_deals']
                        successful_deals_count = kpi_data['successful_deals_count']
                        success_rate = kpi_data['success_rate']
                        avg_deal_value = kpi_data['avg_deal_value']
                        avg_nights = kpi_data['avg_nights']
                        extention_rate = kpi_data['extention_rate']
                        new_customers = kpi_data['new_customers']
                        returning_customers = kpi_data['returning_customers']

                        prev_total_deals = kpi_data['prev_total_deals']
                        prev_successful_deals_count = kpi_data['prev_successful_deals_count']
                        prev_success_rate = kpi_data['prev_success_rate']
                        prev_avg_deal_value = kpi_data['prev_avg_deal_value']
                        prev_avg_nights = kpi_data['prev_avg_nights']
                        prev_extention_rate = kpi_data['prev_extention_rate']
                        prev_new_customers = kpi_data['prev_new_customers']
                        prev_returning_customers = kpi_data['prev_returning_customers']

                        st.markdown("### Key Performance Indicators (KPIs)")
                        colKPI1, colKPI2, colKPI3, colKPI4 = st.columns(4)
                        colKPI1.metric(
                            "Total Deals",
                            f"{total_deals}",
                            pct_diff(total_deals, prev_total_deals)
                        )
                        colKPI2.metric(
                            "Successful Deals",
                            f"{successful_deals_count}",
                            pct_diff(successful_deals_count, prev_successful_deals_count)
                        )
                        colKPI3.metric(
                            "Success Rate (%)",
                            f"{success_rate:.2f}%",
                            pct_diff(success_rate, prev_success_rate)
                        )
                        colKPI4.metric(
                            "Avg. Deal Value",
                            f"{avg_deal_value:,.0f}",
                            pct_diff(avg_deal_value, prev_avg_deal_value)
                        )

                        colKPI5, colKPI6, colKPI7, colKPI8 = st.columns(4)
                        colKPI5.metric(
                            "New Customers",
                            f"{new_customers}",
                            pct_diff(new_customers, prev_new_customers)
                        )
                        colKPI6.metric(
                            "Returning Customers",
                            f"{returning_customers}",
                            pct_diff(returning_customers, prev_returning_customers)
                        )
                        colKPI7.metric(
                            "Avg. Nights",
                            f"{avg_nights:.2f}",
                            pct_diff(avg_nights, prev_avg_nights)
                        )
                        colKPI8.metric(
                            "Extention Rate",
                            f"{extention_rate:.2f}%",
                            pct_diff(extention_rate, prev_extention_rate)
                        )

                        st.write("---")

                        # Outlier Detection
                        st.markdown("**Outlier Detection in Deal Values**")
                        deals_df = channel_data_success[['ارزش معامله','تاریخ انجام معامله','کد دیدار شخص معامله','تعداد شب','نوع خرید','VIP Status']].copy()
                        deals_df.dropna(subset=['ارزش معامله'], inplace=True)
                        if len(deals_df) > 5:
                            q1, q3 = np.percentile(deals_df['ارزش معامله'], [25,75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            outliers = deals_df[(deals_df['ارزش معامله'] < lower_bound) | (deals_df['ارزش معامله'] > upper_bound)]
                            if not outliers.empty:
                                st.write(f"Detected {len(outliers)} outlier deal(s). Below is the table of those outlier deals:")
                                st.write(outliers)
                            else:
                                st.write("No outliers detected in deal values.")
                        else:
                            st.info("Not enough data to detect outliers reliably.")

                        if channel_data_success.empty:
                            st.warning("No successful deals found for the selected channel and date range.")
                        else:
                            # RFM distribution
                            channel_customer_ids = channel_data_success['کد دیدار شخص معامله'].unique()
                            channel_rfm_data = rfm_data[rfm_data['Customer ID'].isin(channel_customer_ids)]
                            if channel_rfm_data.empty:
                                st.warning("No RFM data available for the selected channel and VIP statuses.")
                            else:
                                # Cluster distribution (frequency)
                                cluster_counts = channel_rfm_data['RFM_segment_label'].value_counts().reset_index()
                                cluster_counts.columns = ['RFM_segment_label', 'Count']
                                fig_channel_freq = px.bar(
                                    cluster_counts,
                                    x='RFM_segment_label',
                                    y='Count',
                                    title="Cluster Distribution (Frequency)",
                                    labels={'RFM_segment_label': 'RFM Segment','Count': 'Number of Customers'},
                                    text='Count',
                                    color='RFM_segment_label',
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                fig_channel_freq.update_traces(textposition='outside')
                                st.plotly_chart(fig_channel_freq)

                                # Cluster distribution (monetary)
                                channel_monetary = channel_rfm_data.groupby('RFM_segment_label')['Monetary'].sum().reset_index()
                                fig_channel_monetary = px.bar(
                                    channel_monetary,
                                    x='RFM_segment_label',
                                    y='Monetary',
                                    title="Cluster Distribution (Monetary)",
                                    labels={'RFM_segment_label': 'RFM Segment','Monetary': 'Total Monetary Value'},
                                    text='Monetary',
                                    color='RFM_segment_label',
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                fig_channel_monetary.update_traces(textposition='outside')
                                st.plotly_chart(fig_channel_monetary)

                            st.subheader("Customer Details")
                            channel_nights = channel_data_success.groupby('کد دیدار شخص معامله')['تعداد شب'].sum().reset_index()
                            channel_nights.rename(columns={'کد دیدار شخص معامله': 'Customer ID','تعداد شب': 'Total Nights'}, inplace=True)
                            if 'Customer ID' in rfm_data.columns:
                                customer_details = rfm_data[['Customer ID','First Name','Phone Number','Last Name','VIP Status','Recency','Frequency','Monetary','average stay','Is Monthly','Is staying']].copy()
                                if 'RFM_segment_label' in rfm_data.columns:
                                    customer_details['RFM_segment_label'] = rfm_data['RFM_segment_label']
                            else:
                                customer_details = pd.DataFrame()

                            if not customer_details.empty:
                                customer_details = customer_details.merge(channel_nights, on='Customer ID', how='right').fillna(0)
                            else:
                                customer_details = channel_nights

                            st.write(customer_details)
                            csv_data = convert_df(customer_details)
                            excel_data = convert_df_to_excel(customer_details)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(label="Download data as CSV", data=csv_data, file_name='channel_analysis.csv', mime='text/csv')
                            with col2:
                                st.download_button(label="Download data as Excel", data=excel_data, file_name='channel_analysis.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                            # Time Series
                            st.subheader("Time Series Analysis of Sales")
                            daily_df = st.session_state.single_channel_daily_df
                            if daily_df is None or daily_df.empty:
                                st.info("No time-series data available for this channel.")
                            else:
                                days_in_range = (end_date - start_date).days + 1
                                kpi_options = ['Total Deals','Successful Deals','New Customers','Returning Customers','Average Deal Value','Average Nights']
                                selected_kpis_to_plot = st.multiselect(
                                    "Select KPI(s) to Plot:", 
                                    kpi_options, 
                                    default=['Total Deals','Successful Deals'], 
                                    key='single_channel_ts_kpis'
                                )
                                if selected_kpis_to_plot:
                                    for c in selected_kpis_to_plot:
                                        df_p = daily_df[['Date', c]].copy()
                                        df_p.sort_values('Date', inplace=True)
                                        if days_in_range < 60:
                                            df_p[c+'_7d_MA'] = df_p[c].rolling(7).mean()
                                            fig_ts = px.line(
                                                df_p,
                                                x='Date',
                                                y=[c, c+'_7d_MA'],
                                                title=f"Time Series of {c} (with 7 day MA)",
                                                labels={'value': f"{c}"},
                                                color_discrete_sequence=px.colors.qualitative.Set1
                                            )
                                        else:
                                            df_p[c+'_30d_MA'] = df_p[c].rolling(30).mean()
                                            fig_ts = px.line(
                                                df_p,
                                                x='Date',
                                                y=[c, c+'_30d_MA'],
                                                title=f"Time Series of {c} (with 30 day MA)",
                                                labels={'value': f"{c}"},
                                                color_discrete_sequence=px.colors.qualitative.Set1
                                            )
                                        st.plotly_chart(fig_ts)

                ###########################################################################
                #  COMPARE TWO CHANNELS
                ###########################################################################
                with tabs[1]:
                    st.markdown("### Compare Two Channels")

                    vip_options_compare_two = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_compare_two = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_channel_compare_two')
                    if select_all_vips_compare_two:
                        selected_vips_channel_compare_two = vip_options_compare_two
                    else:
                        selected_vips_channel_compare_two = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_compare_two,
                            default=[],
                            key='vips_multiselect_channel_compare_two'
                        )

                    with st.form(key='compare_two_channels_form', clear_on_submit=False):
                        two_channels = st.multiselect("Select Two Channels:", options=sale_channels_options, max_selections=2, key='two_channels_select')
                        min_date_compare = data['تاریخ انجام معامله'].min()
                        max_date_compare = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date_compare) or pd.isna(max_date_compare):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date_compare = min_date_compare.date()
                        max_date_compare = max_date_compare.date()

                        start_date_compare = st.date_input("Start Date", value=min_date_compare, min_value=min_date_compare, max_value=max_date_compare, key='compare_two_channel_start_date')
                        end_date_compare = st.date_input("End Date", value=max_date_compare, min_value=min_date_compare, max_value=max_date_compare, key='compare_two_channel_end_date')

                        compare_kpi_options = [
                            'Total Deals','Successful Deals','Average Deal Value','Average Nights',
                            'Extension Rate','New Customers','Returning Customers'
                        ]
                        selected_compare_kpis = st.multiselect(
                            "Select KPI(s) to Plot",
                            compare_kpi_options,
                            default=['Average Deal Value','Extension Rate'],
                            key='compare_two_channels_kpis'
                        )
                        apply_compare_two = st.form_submit_button(label='Compare')

                    if "two_channels_results" not in st.session_state:
                        st.session_state.two_channels_results = None
                        st.session_state.two_channels_tsdata = None

                    if apply_compare_two:
                        if len(two_channels) == 2:
                            if selected_vips_channel_compare_two:
                                ch1 = two_channels[0]
                                ch2 = two_channels[1]

                                df_ch1 = data[
                                    (data['شیوه آشنایی معامله'] == ch1) &
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_compare)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_compare)) &
                                    (data['VIP Status'].isin(selected_vips_channel_compare_two))
                                ]
                                df_ch2 = data[
                                    (data['شیوه آشنایی معامله'] == ch2) &
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_compare)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_compare)) &
                                    (data['VIP Status'].isin(selected_vips_channel_compare_two))
                                ]
                                df_ch1_success = df_ch1[df_ch1['وضعیت معامله'] == 'موفق']
                                df_ch2_success = df_ch2[df_ch2['وضعیت معامله'] == 'موفق']

                                def kpi_calc(df_all, df_succ):
                                    td = len(df_all)
                                    sd = len(df_succ)
                                    avgv = df_succ['ارزش معامله'].mean() if not df_succ.empty else 0
                                    nights_ = df_succ['تعداد شب'].mean() if not df_succ.empty else 0
                                    ext = df_succ[df_succ['نوع خرید'] == 'تمدید']
                                    ext_count = len(ext)
                                    ex_rate = (ext_count/sd*100) if sd>0 else 0
                                    nw = 0
                                    rt = 0
                                    if not df_succ.empty:
                                        earliest_global = global_first_deal_date_series_channels.to_dict()
                                        for cid_ in df_succ['کد دیدار شخص معامله'].unique():
                                            fd_ = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fd_):
                                                if start_date_compare <= fd_.date() <= end_date_compare:
                                                    nw += 1
                                                elif fd_.date() < start_date_compare:
                                                    rt += 1
                                    return {
                                        'Channel': (df_all['شیوه آشنایی معامله'].iloc[0] if not df_all.empty else ""),
                                        'Total Deals': td,
                                        'Successful Deals': sd,
                                        'Average Deal Value': avgv,
                                        'Average Nights': nights_,
                                        'Extension Rate': ex_rate,
                                        'New Customers': nw,
                                        'Returning Customers': rt
                                    }

                                ch1_stats = kpi_calc(df_ch1, df_ch1_success)
                                ch2_stats = kpi_calc(df_ch2, df_ch2_success)

                                st.session_state.two_channels_results = (ch1_stats, ch2_stats)

                                if not df_ch1_success.empty:
                                    df_ch1_success = df_ch1_success.copy()
                                    df_ch1_success['Channel'] = ch1
                                if not df_ch2_success.empty:
                                    df_ch2_success = df_ch2_success.copy()
                                    df_ch2_success['Channel'] = ch2
                                combined_success = pd.concat([df_ch1_success, df_ch2_success], ignore_index=True)
                                st.session_state.two_channels_tsdata = combined_success
                            else:
                                st.warning("Please select at least one VIP status.")
                        else:
                            st.warning("Please select exactly two channels.")

                    if st.session_state.two_channels_results is not None:
                        ch1_res, ch2_res = st.session_state.two_channels_results
                        if ch1_res['Channel'] and ch2_res['Channel']:
                            st.markdown("### Comparison KPIs")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**{ch1_res['Channel']}**")
                                st.metric("Total Deals", f"{ch1_res['Total Deals']}")
                                st.metric("Successful Deals", f"{ch1_res['Successful Deals']}")
                                st.metric("Average Deal Value", f"{ch1_res['Average Deal Value']:.0f}")
                                st.metric("Average Nights", f"{ch1_res['Average Nights']:.2f}")
                                st.metric("Extension Rate (%)", f"{ch1_res['Extension Rate']:.2f}%")
                                st.metric("New Customers", f"{ch1_res['New Customers']}")
                                st.metric("Returning Customers", f"{ch1_res['Returning Customers']}")
                            with c2:
                                st.markdown(f"**{ch2_res['Channel']}**")
                                st.metric("Total Deals", f"{ch2_res['Total Deals']}")
                                st.metric("Successful Deals", f"{ch2_res['Successful Deals']}")
                                st.metric("Average Deal Value", f"{ch2_res['Average Deal Value']:.0f}")
                                st.metric("Average Nights", f"{ch2_res['Average Nights']:.2f}")
                                st.metric("Extension Rate (%)", f"{ch2_res['Extension Rate']:.2f}%")
                                st.metric("New Customers", f"{ch2_res['New Customers']}")
                                st.metric("Returning Customers", f"{ch2_res['Returning Customers']}")

                            st.write("---")
                            st.markdown("**Direct Comparison of Each KPI**")

                            # We'll normalize each KPI for side-by-side
                            comp_data = [
                                {'KPI':'Total Deals','Channel':ch1_res['Channel'],'Value': ch1_res['Total Deals']},
                                {'KPI':'Total Deals','Channel':ch2_res['Channel'],'Value': ch2_res['Total Deals']},
                                {'KPI':'Successful Deals','Channel':ch1_res['Channel'],'Value': ch1_res['Successful Deals']},
                                {'KPI':'Successful Deals','Channel':ch2_res['Channel'],'Value': ch2_res['Successful Deals']},
                                {'KPI':'Average Deal Value','Channel':ch1_res['Channel'],'Value': ch1_res['Average Deal Value']},
                                {'KPI':'Average Deal Value','Channel':ch2_res['Channel'],'Value': ch2_res['Average Deal Value']},
                                {'KPI':'Average Nights','Channel':ch1_res['Channel'],'Value': ch1_res['Average Nights']},
                                {'KPI':'Average Nights','Channel':ch2_res['Channel'],'Value': ch2_res['Average Nights']},
                                {'KPI':'Extension Rate','Channel':ch1_res['Channel'],'Value': ch1_res['Extension Rate']},
                                {'KPI':'Extension Rate','Channel':ch2_res['Channel'],'Value': ch2_res['Extension Rate']},
                                {'KPI':'New Customers','Channel':ch1_res['Channel'],'Value': ch1_res['New Customers']},
                                {'KPI':'New Customers','Channel':ch2_res['Channel'],'Value': ch2_res['New Customers']},
                                {'KPI':'Returning Customers','Channel':ch1_res['Channel'],'Value': ch1_res['Returning Customers']},
                                {'KPI':'Returning Customers','Channel':ch2_res['Channel'],'Value': ch2_res['Returning Customers']},
                            ]
                            comp_df_side = pd.DataFrame(comp_data)

                            # For each KPI, normalize
                            comp_df_list = []
                            for kpi_name in comp_df_side['KPI'].unique():
                                sub = comp_df_side[comp_df_side['KPI'] == kpi_name].copy()
                                max_val = sub['Value'].max()
                                if max_val == 0:
                                    sub['Normalized Value'] = 0
                                else:
                                    sub['Normalized Value'] = sub['Value']/max_val
                                comp_df_list.append(sub)
                            comp_df_side_final = pd.concat(comp_df_list, ignore_index=True)

                            fig_kpi_compare = px.bar(
                                comp_df_side_final,
                                x='KPI',
                                y='Normalized Value',
                                color='Channel',
                                barmode='group',
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                title="Side-by-Side KPI Comparison (Normalized)"
                            )
                            fig_kpi_compare.update_traces(
                                hovertemplate='<b>KPI</b>: %{x}<br><b>Channel</b>: %{color}<br>Value: %{customdata[0]}<extra></extra>',
                                customdata=np.expand_dims(comp_df_side_final['Value'], axis=1)
                            )
                            st.plotly_chart(fig_kpi_compare)

                    # Time-Series
                    if (
                        'two_channels_tsdata' in st.session_state and
                        st.session_state.two_channels_tsdata is not None and
                        not st.session_state.two_channels_tsdata.empty and
                        selected_compare_kpis
                    ):
                        ts_df = st.session_state.two_channels_tsdata.copy()
                        ts_df['Date'] = pd.to_datetime(ts_df['تاریخ انجام معامله'], errors='coerce')
                        ts_df.dropna(subset=['Date'], inplace=True)

                        st.markdown("### Time Series Comparison (Each KPI in its own plot)")
                        days_in_range_compare = (end_date_compare - start_date_compare).days + 1

                        existing_channels_in_ts = ts_df['Channel'].unique()

                        # color pairs
                        # fallback if channels not found in mapping
                        def get_channel_colors(chname):
                            color_pairs = {
                                ch1_res['Channel'] if ch1_res else 'ChannelA': ('#d62728','#ffa09e'),  # bold red, pastel red
                                ch2_res['Channel'] if ch2_res else 'ChannelB': ('#1f77b4','#aec7e8'),  # bold blue, pastel
                            }
                            return color_pairs.get(chname, ('#2ca02c','#98df8a'))

                        for k in selected_compare_kpis:
                            day_list = pd.date_range(start=start_date_compare, end=end_date_compare, freq='D')
                            daily_list = []
                            earliest_global = global_first_deal_date_series_channels.to_dict()
                            sub_columns = ts_df[['Date','Channel','کد دیدار شخص معامله','ارزش معامله','نوع خرید','تعداد شب']].copy()

                            for dday in day_list:
                                day_sub = sub_columns[sub_columns['Date'].dt.date == dday.date()]
                                for ch_ in day_sub['Channel'].unique():
                                    sub2 = day_sub[day_sub['Channel'] == ch_]
                                    val = 0
                                    if k == 'Average Deal Value':
                                        val = sub2['ارزش معامله'].sum()/len(sub2) if len(sub2)>0 else 0
                                    elif k == 'Extension Rate':
                                        ex_cnt = len(sub2[sub2['نوع خرید']=='تمدید'])
                                        tot_cnt = len(sub2)
                                        val = (ex_cnt/tot_cnt*100) if tot_cnt>0 else 0
                                    elif k == 'Average Nights':
                                        val = sub2['تعداد شب'].mean() if len(sub2)>0 else 0
                                    elif k == 'Total Deals':
                                        val = len(sub2)
                                    elif k == 'Successful Deals':
                                        val = len(sub2)
                                    elif k == 'New Customers':
                                        newC = 0
                                        for cid_ in sub2['کد دیدار شخص معامله'].unique():
                                            fdate = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fdate) and fdate.date() == dday.date():
                                                newC += 1
                                        val = newC
                                    elif k == 'Returning Customers':
                                        retC = 0
                                        for cid_ in sub2['کد دیدار شخص معامله'].unique():
                                            fdate = earliest_global.get(cid_, pd.NaT)
                                            if not pd.isna(fdate) and fdate.date() < dday.date():
                                                retC += 1
                                        val = retC

                                    daily_list.append({
                                        'Date': dday,
                                        'Channel': ch_,
                                        'Value': val
                                    })
                                # fill missing channel with 0
                                for ch_ in existing_channels_in_ts:
                                    if ch_ not in day_sub['Channel'].unique():
                                        daily_list.append({'Date': dday, 'Channel': ch_, 'Value': 0})

                            daily_k_df = pd.DataFrame(daily_list)
                            daily_k_df.sort_values(['Channel','Date'], inplace=True)

                            if days_in_range_compare < 60:
                                daily_k_df['MA'] = daily_k_df.groupby('Channel')['Value'].transform(lambda x: x.rolling(7).mean())
                                nameMA = '7d MA'
                            else:
                                daily_k_df['MA'] = daily_k_df.groupby('Channel')['Value'].transform(lambda x: x.rolling(30).mean())
                                nameMA = '30d MA'

                            fig_ts = go.Figure()
                            fig_ts.update_layout(
                                title=f"{k} Over Time",
                                xaxis_title="Date",
                                yaxis_title=f"{k}"
                            )

                            for ch_ in daily_k_df['Channel'].unique():
                                sub_ch = daily_k_df[daily_k_df['Channel'] == ch_]
                                raw_color, pastel_color = get_channel_colors(ch_)

                                fig_ts.add_trace(go.Scatter(
                                    x=sub_ch['Date'],
                                    y=sub_ch['Value'],
                                    mode='lines+markers',
                                    name=f"{ch_} - raw {k}",
                                    line=dict(color=raw_color, width=2),
                                    marker=dict(color=raw_color, size=5)
                                ))
                                fig_ts.add_trace(go.Scatter(
                                    x=sub_ch['Date'],
                                    y=sub_ch['MA'],
                                    mode='lines',
                                    name=f"{ch_} - {nameMA}",
                                    line=dict(color=pastel_color, width=3, dash='dot')
                                ))

                            st.plotly_chart(fig_ts)

                ###########################################################################
                #  COMPARE ALL CHANNELS
                ###########################################################################
                with tabs[2]:
                    st.markdown("### Compare All Channels")

                    vip_options_compare_all = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_compare_all = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_channel_compare_all')
                    if select_all_vips_compare_all:
                        selected_vips_channel_compare_all = vip_options_compare_all
                    else:
                        selected_vips_channel_compare_all = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_compare_all,
                            default=[],
                            key='vips_multiselect_channel_compare_all'
                        )

                    with st.form(key='compare_all_channels_form', clear_on_submit=False):
                        min_date_all = data['تاریخ انجام معامله'].min()
                        max_date_all = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date_all) or pd.isna(max_date_all):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date_all = min_date_all.date()
                        max_date_all = max_date_all.date()

                        start_date_all = st.date_input("Start Date", value=min_date_all, min_value=min_date_all, max_value=max_date_all, key='compare_all_channel_start_date')
                        end_date_all = st.date_input("End Date", value=max_date_all, min_value=min_date_all, max_value=max_date_all, key='compare_all_channel_end_date')
                        apply_compare_all = st.form_submit_button(label='Compare All Channels')

                    if "compare_all_channels_results" not in st.session_state:
                        st.session_state.compare_all_channels_results = None

                    if apply_compare_all:
                        if selected_vips_channel_compare_all:
                            all_channels_data = data[
                                (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date_all)) &
                                (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date_all)) &
                                (data['VIP Status'].isin(selected_vips_channel_compare_all))
                            ]
                            if all_channels_data.empty:
                                st.warning("No deals found for the selected VIP statuses in the specified date range.")
                            else:
                                channels_list = all_channels_data['شیوه آشنایی معامله'].dropna().unique().tolist()
                                final_rows = []
                                for ch_ in channels_list:
                                    sel_df = all_channels_data[all_channels_data['شیوه آشنایی معامله'] == ch_]
                                    sel_suc = sel_df[sel_df['وضعیت معامله'] == 'موفق']
                                    td_ = len(sel_df)
                                    sd_ = len(sel_suc)
                                    sr_ = (sd_/td_)*100 if td_>0 else 0
                                    av_ = sel_suc['ارزش معامله'].mean() if not sel_suc.empty else 0
                                    ni_ = sel_suc['تعداد شب'].mean() if not sel_suc.empty else 0
                                    ex_ = sel_suc[sel_suc['نوع خرید'] == 'تمدید']
                                    ex_cnt_ = len(ex_)
                                    ex_rate_ = (ex_cnt_/sd_*100) if sd_>0 else 0
                                    n_c = 0
                                    r_c = 0
                                    if not sel_suc.empty:
                                        for cc in sel_suc['کد دیدار شخص معامله'].unique():
                                            fd = global_first_deal_date_series_channels.get(cc, pd.NaT)
                                            if not pd.isna(fd):
                                                if start_date_all <= fd.date() <= end_date_all:
                                                    n_c += 1
                                                elif fd.date() < start_date_all:
                                                    r_c += 1
                                    final_rows.append({
                                        'Sale Channel': ch_,
                                        'Total Deals': td_,
                                        'Successful Deals': sd_,
                                        'Success Rate': sr_,
                                        'Avg Deal Value': av_,
                                        'Avg Nights': ni_,
                                        'Extension Rate': ex_rate_,
                                        'New Customers': n_c,
                                        'Returning Customers': r_c
                                    })
                                comp_df = pd.DataFrame(final_rows)
                                st.session_state.compare_all_channels_results = comp_df
                        else:
                            st.warning("Please select at least one VIP status.")

                    if st.session_state.compare_all_channels_results is not None and not st.session_state.compare_all_channels_results.empty:
                        comp_df = st.session_state.compare_all_channels_results
                        st.write("### All Channels Comparison")
                        st.write(comp_df)

                        c_csv = convert_df(comp_df)
                        c_excel = convert_df_to_excel(comp_df)
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.download_button(label="Download as CSV", data=c_csv, file_name='all_channels_comparison.csv', mime='text/csv')
                        with cc2:
                            st.download_button(label="Download as Excel", data=c_excel, file_name='all_channels_comparison.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                        if not comp_df.empty:
                            fig_all_channels_sd = px.bar(
                                comp_df,
                                x='Sale Channel',
                                y='Successful Deals',
                                title="Successful Deals by Channel",
                                text='Successful Deals',
                                color='Sale Channel',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_channels_sd.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_channels_sd)

                            fig_all_channels_val = px.bar(
                                comp_df,
                                x='Sale Channel',
                                y='Avg Deal Value',
                                title="Average Deal Value by Channel",
                                text='Avg Deal Value',
                                color='Sale Channel',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_channels_val.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_channels_val)

                            fig_all_channels_sr = px.bar(
                                comp_df,
                                x='Sale Channel',
                                y='Success Rate',
                                title="Success Rate (%) by Channel",
                                text='Success Rate',
                                color='Sale Channel',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_channels_sr.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_channels_sr)

                            fig_all_ext = px.bar(
                                comp_df,
                                x='Sale Channel',
                                y='Extension Rate',
                                title="Extension Rate (%) by Channel",
                                text='Extension Rate',
                                color='Sale Channel',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_all_ext.update_traces(textposition='outside')
                            st.plotly_chart(fig_all_ext)

                            # Show separate box plots for each metric
                            st.markdown("#### Separate Box Plots for Key Metrics")
                            for metric in ['Avg Deal Value','Avg Nights','Success Rate','Extension Rate']:
                                bx_df = comp_df[['Sale Channel', metric]].copy()
                                fig_box = px.box(
                                    bx_df,
                                    x='Sale Channel',
                                    y=metric,
                                    color='Sale Channel',
                                    color_discrete_sequence=px.colors.qualitative.Set1,
                                    title=f"Box Plot of {metric} by Channel"
                                )
                                st.plotly_chart(fig_box)

                ###########################################################################
                #  RFM SALES ANALYSIS (for Channels)
                ###########################################################################
                with tabs[3]:
                    st.markdown("### RFM Sales Analysis by Channel")

                    select_all_clusters_channel = st.checkbox("Select all clusters", value=True, key='select_all_clusters_channel_analysis')
                    if 'RFM_segment_label' in rfm_data.columns:
                        channel_cluster_options = sorted(rfm_data['RFM_segment_label'].unique().tolist())
                    else:
                        channel_cluster_options = []

                    if select_all_clusters_channel:
                        selected_clusters_channel = channel_cluster_options
                    else:
                        selected_clusters_channel = st.multiselect(
                            "Select Clusters:",
                            options=channel_cluster_options,
                            default=[],
                            key='clusters_multiselect_channel_analysis'
                        )

                    vip_options_page_cluster = sorted(rfm_data['VIP Status'].unique())
                    select_all_vips_page_cluster = st.checkbox("Select all VIP statuses", value=True, key='select_all_vips_channel_cluster_analysis')
                    if select_all_vips_page_cluster:
                        selected_vips_channel_cluster = vip_options_page_cluster
                    else:
                        selected_vips_channel_cluster = st.multiselect(
                            "Select VIP Status:",
                            options=vip_options_page_cluster,
                            default=[],
                            key='vips_multiselect_channel_cluster_analysis'
                        )

                    with st.form(key='channel_cluster_form', clear_on_submit=False):
                        min_date = data['تاریخ انجام معامله'].min()
                        max_date = data['تاریخ انجام معامله'].max()
                        if pd.isna(min_date) or pd.isna(max_date):
                            st.warning("Date range is invalid. Please check your data.")
                            st.stop()

                        min_date = min_date.date()
                        max_date = max_date.date()

                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key='channel_cluster_start_date')
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key='channel_cluster_end_date')

                        apply_cluster_filters = st.form_submit_button(label='Apply Filters')

                    if "channel_rfm_sales_data" not in st.session_state:
                        st.session_state.channel_rfm_sales_data = None
                        st.session_state.channel_rfm_sales_kpis = None

                    if apply_cluster_filters:
                        if len(selected_clusters_channel) == 0:
                            st.warning("Please select at least one cluster (or ensure RFM_segment_label is present).")
                            st.session_state.channel_rfm_sales_data = None
                            st.session_state.channel_rfm_sales_kpis = None
                        else:
                            if selected_vips_channel_cluster:
                                date_filtered_data_all = data[
                                    (data['تاریخ انجام معامله'] >= pd.to_datetime(start_date)) &
                                    (data['تاریخ انجام معامله'] <= pd.to_datetime(end_date))
                                ]
                                if 'RFM_segment_label' not in rfm_data.columns:
                                    st.error("RFM_segment_label column not found in rfm_data. Cannot filter by cluster.")
                                    st.session_state.channel_rfm_sales_data = None
                                    st.session_state.channel_rfm_sales_kpis = None
                                else:
                                    cluster_customers = rfm_data[rfm_data['RFM_segment_label'].isin(selected_clusters_channel)]['Customer ID'].unique()
                                    cluster_deals_all = date_filtered_data_all[
                                        date_filtered_data_all['کد دیدار شخص معامله'].isin(cluster_customers) &
                                        date_filtered_data_all['VIP Status'].isin(selected_vips_channel_cluster)
                                    ]
                                    if cluster_deals_all.empty:
                                        st.warning("No deals found for the selected clusters and VIP statuses in the specified date range.")
                                        st.session_state.channel_rfm_sales_data = None
                                        st.session_state.channel_rfm_sales_kpis = None
                                    else:
                                        channel_deals = cluster_deals_all[cluster_deals_all['وضعیت معامله'] == 'موفق']
                                        total_deals = len(cluster_deals_all)
                                        successful_deals_count = len(channel_deals)
                                        success_rate = (successful_deals_count / total_deals)*100 if total_deals>0 else 0
                                        new_customers = 0
                                        returning_customers = 0
                                        if not channel_deals.empty:
                                            unique_customers = channel_deals['کد دیدار شخص معامله'].unique()
                                            for cid in unique_customers:
                                                first_deal_date = global_first_deal_date_series_channels.get(cid, pd.NaT)
                                                if pd.isna(first_deal_date):
                                                    continue
                                                if start_date <= first_deal_date.date() <= end_date:
                                                    new_customers += 1
                                                elif first_deal_date.date() < start_date:
                                                    returning_customers += 1

                                        avg_deal_value = channel_deals['ارزش معامله'].mean() if not channel_deals.empty else 0
                                        avg_nights = channel_deals['تعداد شب'].mean() if not channel_deals.empty else 0
                                        channel_extentions = channel_deals[channel_deals['نوع خرید'] == 'تمدید']
                                        channel_extentions_count = len(channel_extentions)
                                        channel_extention_rate = (channel_extentions_count / successful_deals_count*100) if successful_deals_count>0 else 0

                                        prev_length = (end_date - start_date).days + 1
                                        prev_end = start_date - timedelta(days=1)
                                        prev_start = prev_end - timedelta(days=prev_length - 1)
                                        prev_data_all = data[
                                            (data['تاریخ انجام معامله'] >= pd.to_datetime(prev_start)) &
                                            (data['تاریخ انجام معامله'] <= pd.to_datetime(prev_end))
                                        ]
                                        prev_data_all = prev_data_all[
                                            prev_data_all['کد دیدار شخص معامله'].isin(cluster_customers) &
                                            prev_data_all['VIP Status'].isin(selected_vips_channel_cluster)
                                        ]
                                        prev_deals = prev_data_all[prev_data_all['وضعیت معامله'] == 'موفق']
                                        if not prev_data_all.empty:
                                            ptd = len(prev_data_all)
                                            psd = len(prev_deals)
                                            psr = (psd / ptd)*100 if ptd>0 else 0
                                            pav = prev_deals['ارزش معامله'].mean() if not prev_deals.empty else 0
                                            pni = prev_deals['تعداد شب'].mean() if not prev_deals.empty else 0
                                            pext = prev_deals[prev_deals['نوع خرید'] == 'تمدید']
                                            pext_cnt = len(pext)
                                            pext_rate = (pext_cnt/psd*100) if psd>0 else 0
                                            pnew_c = 0
                                            pret_c = 0
                                            if not prev_deals.empty:
                                                for p_cid in prev_deals['کد دیدار شخص معامله'].unique():
                                                    fd = global_first_deal_date_series_channels.get(p_cid, pd.NaT)
                                                    if not pd.isna(fd):
                                                        if prev_start <= fd.date() <= prev_end:
                                                            pnew_c += 1
                                                        elif fd.date() < prev_start:
                                                            pret_c += 1
                                        else:
                                            ptd = 0
                                            psd = 0
                                            psr = 0
                                            pav = 0
                                            pni = 0
                                            pext_rate = 0
                                            pnew_c = 0
                                            pret_c = 0

                                        st.session_state.channel_rfm_sales_data = channel_deals.copy()
                                        st.session_state.channel_rfm_sales_kpis = {
                                            'total_deals': total_deals,
                                            'successful_deals_count': successful_deals_count,
                                            'success_rate': success_rate,
                                            'avg_deal_value': avg_deal_value,
                                            'avg_nights': avg_nights,
                                            'channel_extention_rate': channel_extention_rate,
                                            'new_customers': new_customers,
                                            'returning_customers': returning_customers,
                                            'ptd': ptd,
                                            'psd': psd,
                                            'psr': psr,
                                            'pav': pav,
                                            'pni': pni,
                                            'pext_rate': pext_rate,
                                            'pnew_c': pnew_c,
                                            'pret_c': pret_c
                                        }
                            else:
                                st.warning("Please select at least one VIP status.")

                    if st.session_state.channel_rfm_sales_data is not None and st.session_state.channel_rfm_sales_kpis is not None:
                        channel_deals = st.session_state.channel_rfm_sales_data
                        kpis = st.session_state.channel_rfm_sales_kpis

                        def pdiff(x, y):
                            if y == 0:
                                return None
                            return f"{((x-y)/abs(y)*100):.2f}%"

                        total_deals = kpis['total_deals']
                        successful_deals_count = kpis['successful_deals_count']
                        success_rate = kpis['success_rate']
                        avg_deal_value = kpis['avg_deal_value']
                        avg_nights = kpis['avg_nights']
                        channel_extention_rate = kpis['channel_extention_rate']
                        new_customers = kpis['new_customers']
                        returning_customers = kpis['returning_customers']
                        ptd = kpis['ptd']
                        psd = kpis['psd']
                        psr = kpis['psr']
                        pav = kpis['pav']
                        pni = kpis['pni']
                        pext_rate = kpis['pext_rate']
                        pnew_c = kpis['pnew_c']
                        pret_c = kpis['pret_c']

                        colKPI1, colKPI2, colKPI3, colKPI4 = st.columns(4)
                        colKPI1.metric(
                            "Total Deals", 
                            f"{total_deals}", 
                            pdiff(total_deals, ptd)
                        )
                        colKPI2.metric(
                            "Successful Deals", 
                            f"{successful_deals_count}",
                            pdiff(successful_deals_count, psd)
                        )
                        colKPI3.metric(
                            "Success Rate (%)",
                            f"{success_rate:.2f}%",
                            pdiff(success_rate, psr)
                        )
                        colKPI4.metric(
                            "Avg. Deal Value",
                            f"{avg_deal_value:,.0f}",
                            pdiff(avg_deal_value, pav)
                        )

                        colKPI5, colKPI6, colKPI7, colKPI8 = st.columns(4)
                        colKPI5.metric(
                            "New Customers",
                            f"{new_customers}",
                            pdiff(new_customers, pnew_c)
                        )
                        colKPI6.metric(
                            "Returning Customers",
                            f"{returning_customers}",
                            pdiff(returning_customers, pret_c)
                        )
                        colKPI7.metric(
                            "Avg. Nights",
                            f"{avg_nights:.2f}",
                            pdiff(avg_nights, pni)
                        )
                        colKPI8.metric(
                            "Extention Rate",
                            f"{channel_extention_rate:.2f}%",
                            pdiff(channel_extention_rate, pext_rate)
                        )

                        st.write("---")
                        if channel_deals.empty:
                            st.warning("No successful deals found for these clusters in the specified date range.")
                        else:
                            seller_counts = channel_deals['مسئول معامله'].value_counts().reset_index()
                            seller_counts.columns = ['Seller','Count']
                            fig_seller_channel_freq = px.bar(
                                seller_counts,
                                x='Seller',
                                y='Count',
                                title="Seller Distribution (Frequency)",
                                labels={'Seller': 'Seller','Count': 'Number of Deals'},
                                text='Count',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_seller_channel_freq.update_traces(textposition='outside')
                            st.plotly_chart(fig_seller_channel_freq)

                            seller_monetary = channel_deals.groupby('مسئول معامله')['ارزش معامله'].sum().reset_index()
                            seller_monetary.columns = ['Seller','Monetary']
                            fig_seller_channel_monetary = px.bar(
                                seller_monetary,
                                x='Seller',
                                y='Monetary',
                                title="Seller Distribution (Monetary)",
                                labels={'Seller': 'Seller','Monetary': 'Total Monetary Value'},
                                text='Monetary',
                                color='Seller',
                                color_discrete_sequence=px.colors.qualitative.Set1
                            )
                            fig_seller_channel_monetary.update_traces(textposition='outside')
                            st.plotly_chart(fig_seller_channel_monetary)

                            st.subheader("Successful Deals")
                            if 'RFM_segment_label' in rfm_data.columns:
                                channel_deals = channel_deals.merge(
                                    rfm_data[['Customer ID','RFM_segment_label']],
                                    left_on='کد دیدار شخص معامله',
                                    right_on='Customer ID',
                                    how='left'
                                )
                            if 'RFM_segment_label' in channel_deals.columns:
                                deals_table = channel_deals[[
                                    'Customer ID','نام شخص معامله','نام خانوادگی شخص معامله',
                                    'موبایل شخص معامله','VIP Status','RFM_segment_label',
                                    'مسئول معامله','تعداد شب','ارزش معامله','تاریخ انجام معامله'
                                ]]
                            else:
                                deals_table = channel_deals

                            st.write(deals_table)
                            csv_data = convert_df(deals_table)
                            excel_data = convert_df_to_excel(deals_table)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(label="Download data as CSV", data=csv_data, file_name='channel_cluster_deals.csv', mime='text/csv')
                            with col2:
                                st.download_button(label="Download data as Excel", data=excel_data, file_name='channel_cluster_deals.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                            st.subheader("Time Series Analysis of Sales")
                            channel_deals_time_df = channel_deals[['تاریخ انجام معامله','ارزش معامله']].copy()
                            channel_deals_time_df['تاریخ انجام معامله'] = pd.to_datetime(channel_deals_time_df['تاریخ انجام معامله'], errors='coerce')
                            channel_deals_time_df.dropna(subset=['تاریخ انجام معامله'], inplace=True)
                            if channel_deals_time_df.empty:
                                st.info("No time-series data available for these clusters.")
                            else:
                                channel_deals_time_df = channel_deals_time_df.groupby(channel_deals_time_df['تاریخ انجام معامله'].dt.date)['ارزش معامله'].sum().reset_index()
                                channel_deals_time_df.rename(columns={'تاریخ انجام معامله': 'Date','ارزش معامله': 'Sales'}, inplace=True)
                                channel_deals_time_df['Date'] = pd.to_datetime(channel_deals_time_df['Date'])
                                channel_deals_time_df.sort_values('Date', inplace=True)

                                days_in_rfm = (end_date - start_date).days + 1
                                if days_in_rfm < 60:
                                    channel_deals_time_df['7d_MA'] = channel_deals_time_df['Sales'].rolling(7).mean()
                                    lines_to_use = ['Sales','7d_MA']
                                    chart_title = "Daily Sales Over Time (with 7-day MA)"
                                else:
                                    channel_deals_time_df['30d_MA'] = channel_deals_time_df['Sales'].rolling(30).mean()
                                    lines_to_use = ['Sales','30d_MA']
                                    chart_title = "Daily Sales Over Time (with 30-day MA)"

                                fig_channel_time = px.line(
                                    channel_deals_time_df,
                                    x='Date',
                                    y=lines_to_use,
                                    labels={'value': 'Sales Amount'},
                                    title=chart_title,
                                    color_discrete_sequence=px.colors.qualitative.Set1
                                )
                                st.plotly_chart(fig_channel_time)

                                monthly_df = channel_deals_time_df[['Date','Sales']].copy()
                                monthly_df['Month'] = monthly_df['Date'].dt.to_period('M')
                                monthly_avg = monthly_df.groupby('Month')['Sales'].mean().reset_index()
                                monthly_avg['Month'] = monthly_avg['Month'].astype(str)
                                fig_channel_monthly = px.bar(
                                    monthly_avg,
                                    x='Month',
                                    y='Sales',
                                    labels={'Sales': 'Average Sales'},
                                    title="Monthly Average Sales"
                                )
                                st.plotly_chart(fig_channel_monthly)

                                total_sales_time = channel_deals_time_df['Sales'].sum()
                                avg_sales_time = channel_deals_time_df['Sales'].mean()
                                colA, colB = st.columns(2)
                                with colA:
                                    st.metric("Total Sales (Selected Period)", f"{total_sales_time:,.0f}")
                                with colB:
                                    st.metric("Avg Daily Sales (Selected Period)", f"{avg_sales_time:,.2f}")

                                    ######################
                    # New Channel Transitions Tab
                    ######################
                    with tabs[4]:
                        st.markdown("### Channel Transitions")

                        # 1) UI for selecting the initial channel and RFM clusters
                        chosen_channel = st.selectbox(
                            "Select a Sale Channel (First Reservation Channel)",
                            options=sale_channels_options
                        )

                        rfm_cluster_options = sorted(rfm_data['RFM_segment_label'].dropna().unique())
                        chosen_clusters = st.multiselect(
                            "Select RFM Clusters",
                            options=rfm_cluster_options,
                            default=rfm_cluster_options
                        )

                        if chosen_channel and chosen_clusters:
                            # Prepare the data of successful deals
                            df_success = data[data['وضعیت معامله'] == 'موفق'].copy()
                            df_success = df_success.sort_values("تاریخ انجام معامله")

                            # Find each customer's earliest successful deal
                            first_deals = (
                                df_success.groupby("کد دیدار شخص معامله")
                                .head(1)
                                .reset_index(drop=True)
                            )

                            # 1) Filter to customers whose FIRST reservation was on the chosen channel
                            #    and also whose RFM cluster is in the chosen set
                            first_channel_customers = first_deals[
                                first_deals["شیوه آشنایی معامله"] == chosen_channel
                            ]["کد دیدار شخص معامله"].unique()

                            # Filter by chosen RFM clusters
                            # We look up the cluster in rfm_data where rfm_data["Customer ID"] == person's code
                            cluster_matched_customers = rfm_data[
                                (rfm_data["Customer ID"].isin(first_channel_customers))
                                & (rfm_data["RFM_segment_label"].isin(chosen_clusters))
                            ]["Customer ID"].unique()

                            if len(cluster_matched_customers) == 0:
                                st.warning("No customers found matching both the selected channel and these RFM clusters.")
                            else:
                                # ----------  Part 1: Next reservations and their channels  ----------
                                # We want subsequent deals (beyond the first) for these customers
                                subsequent_deals = df_success[df_success["کد دیدار شخص معامله"].isin(cluster_matched_customers)].copy()

                                # Attach earliest deal date so we can filter out the first deal
                                earliest_dates = (
                                    df_success.groupby("کد دیدار شخص معامله")["تاریخ انجام معامله"]
                                    .min()
                                    .rename("EarliestDealDate")
                                )
                                subsequent_deals = subsequent_deals.merge(
                                    earliest_dates,
                                    left_on="کد دیدار شخص معامله",
                                    right_index=True
                                )

                                # Keep only deals strictly AFTER the first deal date
                                subsequent_deals = subsequent_deals[
                                    subsequent_deals["تاریخ انجام معامله"] > subsequent_deals["EarliestDealDate"]
                                ]

                                if subsequent_deals.empty:
                                    st.info("No subsequent reservations found for those customers.")
                                else:
                                    # Count how many next reservations happened on each channel
                                    channel_counts = subsequent_deals["شیوه آشنایی معامله"].value_counts().reset_index()
                                    channel_counts.columns = ["Sale Channel", "Count"]

                                    st.subheader("1) Next Reservations: Which channels were used?")
                                    fig_next_reservations = px.bar(
                                        channel_counts,
                                        x="Sale Channel",
                                        y="Count",
                                        title="Subsequent Reservations by Channel",
                                        text="Count",
                                        labels={"Count": "Number of Non-First Reservations"}
                                    )
                                    fig_next_reservations.update_traces(textposition='outside')
                                    st.plotly_chart(fig_next_reservations)

                                # ----------  Part 2: Customers’ Favorite Reservation Channel  ----------
                                # For the same group of cluster-matched customers, figure out
                                # which channel each one used the most across ALL successful deals
                                # (including the first and subsequent).

                                all_deals_for_these_customers = df_success[
                                    df_success["کد دیدار شخص معامله"].isin(cluster_matched_customers)
                                ].copy()

                                # Group (customer, channel) => count
                                cust_channel_counts = (
                                    all_deals_for_these_customers
                                    .groupby(["کد دیدار شخص معامله", "شیوه آشنایی معامله"])
                                    .size()
                                    .reset_index(name="NumReservations")
                                )
                                # Sort so highest count is first, then drop duplicates
                                cust_channel_counts.sort_values(
                                    ["کد دیدار شخص معامله", "NumReservations"],
                                    ascending=[True, False],
                                    inplace=True
                                )
                                favorite_channels = cust_channel_counts.drop_duplicates(
                                    subset=["کد دیدار شخص معامله"], keep="first"
                                )
                                favorite_channels.rename(columns={"شیوه آشنایی معامله": "FavoriteSaleChannel"}, inplace=True)

                                # Summarize how many times each channel is "favorite"
                                fav_counts = favorite_channels["FavoriteSaleChannel"].value_counts().reset_index()
                                fav_counts.columns = ["Sale Channel", "Count"]

                                st.subheader("2) Favorite Reservation Channel")
                                colA, colB = st.columns([1,1.3])

                                with colA:
                                    st.markdown("#### Column Chart")
                                    fig_fav_channels = px.bar(
                                        fav_counts,
                                        x="Sale Channel",
                                        y="Count",
                                        text="Count",
                                        title="Customers' Favorite Channel (Count)"
                                    )
                                    fig_fav_channels.update_traces(textposition='outside')
                                    st.plotly_chart(fig_fav_channels)

                                with colB:
                                    st.markdown("#### Detailed Table")

                                    # Merge back to RFM data to get user info
                                    detailed_fav = favorite_channels.merge(
                                        rfm_data,
                                        left_on="کد دیدار شخص معامله",
                                        right_on="Customer ID",
                                        how="left"
                                    )

                                    # Pick relevant columns
                                    columns_to_show = [
                                        "Customer ID",
                                        "First Name",
                                        "Last Name",
                                        "Phone Number",
                                        "VIP Status",
                                        "RFM_segment_label",
                                        "Recency",
                                        "Frequency",
                                        "Monetary",
                                        "Total Nights",
                                        "FavoriteSaleChannel"
                                    ]

                                    # Check if 'Total Nights' is in rfm_data (depends on your code)
                                    # If not, handle gracefully:
                                    if "Total Nights" not in detailed_fav.columns:
                                        # you might have "Total Nights" under a different name
                                        # or you can calculate it from deals if you want
                                        # For now, we add a placeholder if missing:
                                        detailed_fav["Total Nights"] = None

                                    final_table = detailed_fav[columns_to_show].copy()

                                    st.dataframe(final_table)

                                    # Download buttons
                                    csv_data = convert_df(final_table)
                                    excel_data = convert_df_to_excel(final_table)

                                    c1, c2 = st.columns(2)
                                    with c1:
                                        st.download_button(
                                            label="Download (CSV)",
                                            data=csv_data,
                                            file_name="favorite_channels.csv",
                                            mime="text/csv"
                                        )
                                    with c2:
                                        st.download_button(
                                            label="Download (Excel)",
                                            data=excel_data,
                                            file_name="favorite_channels.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )

                                # ----------  Part 3: Additional Interesting Data / Charts  ----------
                                # ----------  Part 3: Additional Channel Transition Insights  ----------
                                st.subheader("3) Additional Channel Transition Insights")

                                # We'll create data about how many times the chosen channel is the "from" side vs. the "to" side 
                                # of a reservation transition (in consecutive deals) among cluster_matched_customers.

                                df_cluster_success = df_success[df_success["کد دیدار شخص معامله"].isin(cluster_matched_customers)].copy()
                                df_cluster_success = df_cluster_success.sort_values(["کد دیدار شخص معامله", "تاریخ انجام معامله"])

                                # We will gather all consecutive (channel_i -> channel_(i+1)) transitions
                                # for these cluster-matched customers.
                                transitions = []
                                for cust_id, group_df in df_cluster_success.groupby("کد دیدار شخص معامله"):
                                    group_df = group_df.reset_index(drop=True)
                                    for i in range(len(group_df) - 1):
                                        from_channel = group_df.loc[i, "شیوه آشنایی معامله"]
                                        to_channel   = group_df.loc[i+1, "شیوه آشنایی معامله"]
                                        if pd.notna(from_channel) and pd.notna(to_channel) and from_channel != "" and to_channel != "":
                                            transitions.append((from_channel, to_channel))

                                if not transitions:
                                    st.info("No consecutive channel-to-channel transitions found among these customers.")
                                else:
                                    from collections import Counter, defaultdict
                                    
                                    # Count the total transitions for each (from -> to) pair
                                    transition_counts = Counter(transitions)

                                    # 1) Incoming transitions to the chosen channel
                                    #    i.e., (X -> chosen_channel)
                                    incoming_counts = defaultdict(int)
                                    # 2) Outgoing transitions from the chosen channel
                                    #    i.e., (chosen_channel -> X)
                                    outgoing_counts = defaultdict(int)

                                    for (from_c, to_c), cnt in transition_counts.items():
                                        if to_c == chosen_channel:
                                            incoming_counts[from_c] += cnt
                                        if from_c == chosen_channel:
                                            outgoing_counts[to_c] += cnt

                                    # -------------
                                    #   INCOMING
                                    # -------------
                                    if len(incoming_counts) > 0:
                                        st.markdown(f"#### Incoming Transitions into {chosen_channel}")
                                        incoming_df = pd.DataFrame(
                                            {"From Channel": list(incoming_counts.keys()),
                                            "Count": list(incoming_counts.values())}
                                        ).sort_values("Count", ascending=False)
                                        fig_incoming = px.bar(
                                            incoming_df,
                                            x="From Channel",
                                            y="Count",
                                            text="Count",
                                            labels={"Count": "Number of Transitions"},
                                            title=f"Incoming Transitions to {chosen_channel}"
                                        )
                                        fig_incoming.update_traces(textposition='outside')
                                        st.plotly_chart(fig_incoming)
                                        st.dataframe(incoming_df)
                                    else:
                                        st.info(f"No incoming transitions from other channels to {chosen_channel} among these customers.")

                                    # -------------
                                    #   OUTGOING
                                    # -------------
                                    if len(outgoing_counts) > 0:
                                        st.markdown(f"#### Outgoing Transitions from {chosen_channel}")
                                        outgoing_df = pd.DataFrame(
                                            {"To Channel": list(outgoing_counts.keys()),
                                            "Count": list(outgoing_counts.values())}
                                        ).sort_values("Count", ascending=False)
                                        fig_outgoing = px.bar(
                                            outgoing_df,
                                            x="To Channel",
                                            y="Count",
                                            text="Count",
                                            labels={"Count": "Number of Transitions"},
                                            title=f"Outgoing Transitions from {chosen_channel}"
                                        )
                                        fig_outgoing.update_traces(textposition='outside')
                                        st.plotly_chart(fig_outgoing)
                                        st.dataframe(outgoing_df)
                                    else:
                                        st.info(f"No outgoing transitions from {chosen_channel} to other channels among these customers.")

                                    # -------------
                                    # Net Flow by Other Channels
                                    # (i.e., transitions_in - transitions_out with respect to the chosen channel)
                                    # For each "other channel" X:
                                    #     in_from_X = X -> chosen_channel
                                    #     out_to_X  = chosen_channel -> X
                                    #     net_flow  = in_from_X - out_to_X
                                    # A positive net_flow means more transitions from X into chosen_channel 
                                    #      than from chosen_channel to X. 
                                    # A negative net_flow means the opposite.
                                    # -------------

                                    # Gather all unique channels that appear in either incoming_counts or outgoing_counts
                                    all_involved_channels = set(incoming_counts.keys()) | set(outgoing_counts.keys())

                                    net_rows = []
                                    for ch in sorted(all_involved_channels):
                                        in_val = incoming_counts[ch]
                                        out_val = outgoing_counts[ch]
                                        net_flow = in_val - out_val
                                        net_rows.append({
                                            "Channel": ch,
                                            f"{ch} -> {chosen_channel}": in_val,
                                            f"{chosen_channel} -> {ch}": out_val,
                                            "Net Flow (In - Out)": net_flow
                                        })

                                    if net_rows:
                                        st.markdown(f"#### Net Transitions (In - Out) relative to {chosen_channel}")
                                        net_df = pd.DataFrame(net_rows)
                                        st.dataframe(net_df)

                                        fig_net_flow = px.bar(
                                            net_df,
                                            x="Channel",
                                            y="Net Flow (In - Out)",
                                            text="Net Flow (In - Out)",
                                            labels={"Net Flow (In - Out)": "In - Out"},
                                            title=f"Net Flow (In - Out) with respect to {chosen_channel}"
                                        )
                                        fig_net_flow.update_traces(textposition='outside')
                                        st.plotly_chart(fig_net_flow)
                                    else:
                                        st.info(f"No transitions found when calculating Net Flow for {chosen_channel}.")


            elif page == "Arrival Analysis":
                st.subheader("Arrival Analysis")

                # 1) --- DATE RANGE FILTER (like the rest of the dashboard) ---

                # Ensure 'تاریخ ورود' is a proper datetime column
                data['تاریخ ورود'] = pd.to_datetime(data['تاریخ ورود'], errors='coerce')
                # Drop rows with no arrival date
                df_arrivals = data.dropna(subset=['تاریخ ورود']).copy()

                if df_arrivals.empty:
                    st.warning("No valid arrival dates found in the dataset.")
                    st.stop()

                # Get the min/max arrival dates from the data
                min_date_dt = df_arrivals['تاریخ ورود'].min()
                max_date_dt = df_arrivals['تاریخ ورود'].max()

                if pd.isna(min_date_dt) or pd.isna(max_date_dt):
                    st.warning("Date range is invalid. Please check your data.")
                    st.stop()

                min_date = min_date_dt.date()
                max_date = max_date_dt.date()

                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start of arrival date range",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                with col2:
                    end_date = st.date_input(
                        "End of arrival date range",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )

                if start_date > end_date:
                    st.error("Start date cannot be after end date.")
                    st.stop()

                # 2) --- FILTERS ON COMPLEXES AND HOUSE TYPES (DEPENDENT) ---

                # Complex filter
                complex_options = sorted(df_arrivals['Complex'].dropna().unique().tolist())
                select_all_complexes = st.checkbox("Select all complexes", value=True)
                if select_all_complexes:
                    selected_complexes = complex_options
                else:
                    selected_complexes = st.multiselect(
                        "Select complexes:",
                        options=complex_options,
                        default=[]
                    )
                if not selected_complexes:
                    st.warning("No complexes selected. Showing all by default.")
                    selected_complexes = complex_options

                # Narrow down product options only to what's in the chosen complexes:
                temp_for_complex = df_arrivals[df_arrivals['Complex'].isin(selected_complexes)]
                product_options = sorted(temp_for_complex['عنوان محصول'].dropna().unique().tolist())

                # House type (product) filter
                select_all_products = st.checkbox("Select all house types", value=True)
                if select_all_products:
                    selected_products = product_options
                else:
                    selected_products = st.multiselect(
                        "Select house types:",
                        options=product_options,
                        default=[]
                    )
                if not selected_products:
                    st.warning("No house types selected. Showing all by default.")
                    selected_products = product_options

                # 3) --- APPLY ALL FILTERS ---
                mask = (
                    (df_arrivals['تاریخ ورود'].dt.date >= start_date) &
                    (df_arrivals['تاریخ ورود'].dt.date <= end_date) &
                    (df_arrivals['Complex'].isin(selected_complexes)) &
                    (df_arrivals['عنوان محصول'].isin(selected_products))
                )
                filtered_df = df_arrivals[mask].copy()

                if filtered_df.empty:
                    st.warning("No arrivals found for the selected date range and filters.")
                    st.stop()

                # 4) --- COMPUTE THE METRICS FOR SCOREBOARD ---

                # 4.1) Total Arrivals
                total_arrivals = len(filtered_df)

                # 4.2) Average Weekly Arrivals
                date_range_days = (end_date - start_date).days + 1
                weeks_in_range = date_range_days / 7.0  # approximate
                if weeks_in_range > 0:
                    avg_weekly = total_arrivals / weeks_in_range
                else:
                    avg_weekly = 0

                # 4.3) Average Monthly Arrivals (approx by ~30.44 days/month)
                months_in_range = date_range_days / 30.44
                if months_in_range > 0:
                    avg_monthly = total_arrivals / months_in_range
                else:
                    avg_monthly = 0

                # 4.4) Average Length of Stay
                if 'تعداد شب' in filtered_df.columns:
                    avg_stay = filtered_df['تعداد شب'].mean()
                else:
                    avg_stay = 0

                # 4.5) Extensions count => "نوع خرید" == "تمدید"
                filtered_df['IsExtension'] = filtered_df['نوع خرید'].eq('تمدید')
                total_extensions = filtered_df['IsExtension'].sum()

                # 4.6) New arrivals = non-extensions
                total_new_arrivals = len(filtered_df[~filtered_df['IsExtension']])

                # 5) --- SCOREBOARD DISPLAY ---
                colA1, colA2, colA3 = st.columns(3)
                colA1.metric("Total Arrivals", f"{total_arrivals}")
                colA2.metric("Avg Weekly Arrivals", f"{avg_weekly:.2f}")
                colA3.metric("Avg Monthly Arrivals", f"{avg_monthly:.2f}")

                colB1, colB2, colB3 = st.columns(3)
                colB1.metric("Average Stay (Nights)", f"{avg_stay:.2f}")
                colB2.metric("Total Extensions", f"{total_extensions}")
                colB3.metric("Total New Arrivals", f"{total_new_arrivals}")

                st.write("---")

                # 6) --- TABLE BREAKDOWN BY HOUSE TYPE (عنوان محصول) ---
                st.subheader("Arrival Breakdown by House Type")

                grouped = filtered_df.groupby('عنوان محصول', dropna=False)

                house_type_data = []
                for house_type, subdf in grouped:
                    arrivals_count = len(subdf)
                    avg_stay_ht = subdf['تعداد شب'].mean() if 'تعداد شب' in subdf.columns else 0
                    ext_count = subdf['IsExtension'].sum()
                    new_count = len(subdf[~subdf['IsExtension']])

                    house_type_data.append({
                        'House Type': house_type,
                        'Arrivals': arrivals_count,
                        'Avg Stay': round(avg_stay_ht, 2),
                        'Extensions': ext_count,
                        'New Arrivals': new_count,
                    })

                df_house_type = pd.DataFrame(house_type_data)
                st.dataframe(df_house_type)

                csv_house_type = convert_df(df_house_type)
                excel_house_type = convert_df_to_excel(df_house_type)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        label="Download CSV",
                        data=csv_house_type,
                        file_name="arrival_by_house_type.csv",
                        mime="text/csv"
                    )
                with c2:
                    st.download_button(
                        label="Download Excel",
                        data=excel_house_type,
                        file_name="arrival_by_house_type.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                # 7) --- MONTHLY COLUMN CHARTS FOR EACH COMPLEX ---
                # Each chart is a stacked bar of extension vs new arrivals, one bar per month.
                # Show the sub-segment counts *and* a total label on top of each stacked bar.

                st.write("---")
                st.subheader("Monthly Arrivals by Complex (Extensions vs. New)")

                # Create a 'Month' column (e.g. '2023-07') for grouping
                filtered_df['Month'] = filtered_df['تاریخ ورود'].dt.to_period('M').astype(str)

                # We'll loop over each chosen complex and show a stacked column chart
                for cx in selected_complexes:
                    sub_df = filtered_df[filtered_df['Complex'] == cx]
                    if sub_df.empty:
                        continue  # skip if no data for this complex

                    # Group by Month + IsExtension to get counts
                    monthly_counts = sub_df.groupby(['Month', 'IsExtension']).size().reset_index(name='ArrivalsCount')
                    # Also get monthly totals (regardless of extension)
                    monthly_totals = sub_df.groupby('Month').size().reset_index(name='TotalCount')

                    # Plot a stacked bar chart using Plotly Express
                    fig = px.bar(
                        monthly_counts,
                        x='Month',
                        y='ArrivalsCount',
                        color='IsExtension',  # True/False
                        barmode='stack',
                        title=f"Monthly Arrivals - {cx}",
                        text='ArrivalsCount'
                    )

                    # Position the sub-segment labels inside or outside
                    fig.update_traces(textposition='inside')
                    fig.update_layout(
                        xaxis_title="Month",
                        yaxis_title="Number of Arrivals",
                        # Some spacing so top labels don't get cut off
                        margin=dict(t=80)
                    )

                    # Add an annotation with the total on top of each stacked column
                    for _, row in monthly_totals.iterrows():
                        fig.add_annotation(
                            x=row['Month'],
                            y=row['TotalCount'],
                            text=str(row['TotalCount']),
                            showarrow=False,
                            font=dict(color='black', size=12),
                            xanchor='center',
                            yanchor='bottom'
                        )

                    st.plotly_chart(fig, use_container_width=True)

                st.success("Arrival analysis completed.")



            elif page == 'VIP Analysis':
                st.subheader("VIP Analysis")

                # VIP Filter
                vip_options_page = sorted(rfm_data['VIP Status'].unique())
                default_vips = [vip for vip in vip_options_page if vip != 'Non-VIP']

                select_all_vips_page = st.checkbox("Select all VIP statuses", value=False)
                if select_all_vips_page:
                    selected_vips_vip_analysis = vip_options_page
                else:
                    selected_vips_vip_analysis = st.multiselect(
                        "Select VIP Status:",
                        options=vip_options_page,
                        default=default_vips
                    )

                # Date Range Input
                min_date = data['تاریخ انجام معامله'].min().date()
                max_date = data['تاریخ انجام معامله'].max().date()

                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

                if not selected_vips_vip_analysis:
                    st.warning("Please select at least one VIP status.")
                else:
                    # Filter data
                    date_filtered_data = filtered_data[
                        (filtered_data['تاریخ انجام معامله'].dt.date >= start_date) &
                        (filtered_data['تاریخ انجام معامله'].dt.date <= end_date) &
                        (filtered_data['وضعیت معامله'] == 'موفق') &
                        (filtered_data['VIP Status'].isin(selected_vips_vip_analysis))
                    ]

                    if date_filtered_data.empty:
                        st.warning("No successful deals found for the selected VIP statuses in the specified date range.")
                    else:
                        # Get VIP RFM data
                        vip_customer_ids = date_filtered_data['کد دیدار شخص معامله'].unique()
                        vip_rfm_data = rfm_data[rfm_data['Customer ID'].isin(vip_customer_ids)]

                        if vip_rfm_data.empty:
                            st.warning("No RFM data available for the selected VIP statuses.")
                        else:
                            # Insights
                            filtered_vip_data = vip_rfm_data[vip_rfm_data['VIP Status'] != 'Non-VIP']
                            total_vip_customers = filtered_vip_data['Customer ID'].nunique()
                            total_vip_champions = filtered_vip_data[filtered_vip_data['RFM_segment_label'] == 'Champions']['Customer ID'].nunique()
                            total_vip_non_champions = total_vip_customers - total_vip_champions


                            st.write(f"**Total VIP Customers:** {total_vip_customers}")
                            st.write(f"**Total VIP Champions:** {total_vip_champions}")
                            st.write(f"**Total VIP Non-Champions:** {total_vip_non_champions}")

                            # Number of Champions who are not VIP
                            total_champions_all = rfm_data_filtered_global[rfm_data_filtered_global['RFM_segment_label'] == 'Champions']['Customer ID'].nunique()
                            champions_not_vip = total_champions_all - total_vip_champions
                            st.write(f"**Number of Champions who are not VIP:** {champions_not_vip}")

                            # Plot distribution of VIPs across segments
                            vip_segment_distribution = vip_rfm_data['RFM_segment_label'].value_counts().reset_index()
                            vip_segment_distribution.columns = ['RFM_segment_label', 'Count']

                            fig_vip_segments = px.pie(
                                vip_segment_distribution,
                                names='RFM_segment_label',
                                values='Count',
                                color='RFM_segment_label',
                                color_discrete_map=COLOR_MAP,
                                hole=0.4,
                                title='VIP Customers Distribution across RFM Segments'
                            )

                            fig_vip_segments.update_traces(textposition='inside', textinfo='percent+label')

                            st.plotly_chart(fig_vip_segments)

                            # Additional Insights
                            st.subheader("Additional Insights")

                            # Average Monetary Value per VIP Level
                            avg_monetary_vip = vip_rfm_data.groupby('VIP Status')['Monetary'].mean().reset_index()
                            fig_avg_monetary = px.bar(
                                avg_monetary_vip,
                                x='VIP Status',
                                y='Monetary',
                                title='Average Monetary Value per VIP Level',
                                labels={'Monetary': 'Average Monetary Value'},
                                text='Monetary'
                            )
                            fig_avg_monetary.update_traces(textposition='outside')
                            st.plotly_chart(fig_avg_monetary)

                            # Recency Distribution
                            fig_recency_vip = px.histogram(
                                vip_rfm_data,
                                x='Recency',
                                nbins=50,
                                title='Recency Distribution for VIP Customers',
                                color='VIP Status',
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_recency_vip)

                            # Frequency Distribution
                            fig_frequency_vip = px.histogram(
                                vip_rfm_data,
                                x='Frequency',
                                nbins=20,
                                title='Frequency Distribution for VIP Customers',
                                color='VIP Status',
                                barmode='overlay'
                            )
                            st.plotly_chart(fig_frequency_vip)

                            # ------------------ Customer Table with Editable VIP Status ------------------

                            st.subheader("Edit VIPs")

                            # Prepare customer details
                            vip_customer_details = vip_rfm_data[['Customer ID', 'First Name', 'Last Name', 'VIP Status', 'Phone Number', 'Recency', 'Frequency', 'Monetary', 'RFM_segment_label']].copy()
                            vip_customer_details['First Name'] = vip_customer_details['First Name'].fillna('')
                            vip_customer_details['Phone Number'] = vip_customer_details['Phone Number'].fillna('')
                            vip_customer_details['Last Name'] = vip_customer_details['Last Name'].fillna('')

                            # Add 'New VIP Status' column
                            vip_customer_details['New VIP Status'] = vip_customer_details['VIP Status']

                            # Configure AgGrid
                            vip_status_options = ['Gold VIP', 'Silver VIP', 'Bronze VIP', 'Non-VIP']
                            gb = GridOptionsBuilder.from_dataframe(vip_customer_details)
                            gb.configure_pagination()
                            gb.configure_default_column(editable=False)
                            gb.configure_column('New VIP Status', editable=True, cellEditor='agSelectCellEditor', cellEditorParams={'values': vip_status_options})
                            grid_options = gb.build()

                            grid_response = AgGrid(
                                vip_customer_details,
                                gridOptions=grid_options,
                                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                                update_mode=GridUpdateMode.VALUE_CHANGED,
                                fit_columns_on_grid_load=True
                            )

                            edited_df = grid_response['data']

                            # Password Input
                            password = st.text_input('Enter password to apply changes:', type='password')

                            # Apply Changes Button
                            if st.button('APPLY CHANGES'):
                                 if password != st.secrets["change_password"]:
                                    st.error('Incorrect password.')
                                else:
                                    changed_vip_customers = edited_df[edited_df['VIP Status'] != edited_df['New VIP Status']]
                                    if changed_vip_customers.empty:
                                        st.info('No changes detected.')
                                    else:
                                        # Apply changes
                                        for idx, row in changed_vip_customers.iterrows():
                                            customer_id = row['Customer ID']
                                            phone_number = row['Phone Number']
                                            old_vip_status = row['VIP Status']
                                            new_vip_status = row['New VIP Status']
                                            last_name = row['Last Name']

                                            updated_last_name = update_last_name(last_name, new_vip_status)

                                            # Update local dataframes
                                            edited_df.at[idx, 'Last Name'] = updated_last_name
                                            edited_df.at[idx, 'VIP Status'] = new_vip_status
                                            edited_df.at[idx, 'Last Name'] = updated_last_name
                                            vip_rfm_data.loc[vip_rfm_data['Customer ID'] == customer_id, 'Last Name'] = updated_last_name
                                            vip_rfm_data.loc[vip_rfm_data['Customer ID'] == customer_id, 'VIP Status'] = new_vip_status
                                            rfm_data.loc[rfm_data['Customer ID'] == customer_id, 'Last Name'] = updated_last_name
                                            rfm_data.loc[rfm_data['Customer ID'] == customer_id, 'VIP Status'] = new_vip_status
                                         
                                            # Update via API
                                            success = update_contact_last_name(phone_number, updated_last_name)
                                            customer_name = f"{row['First Name']} {updated_last_name}".strip()
                                            if success:
                                                st.success(f"Customer {customer_name} updated successfully.")
                                            else:
                                                st.error(f"Failed to update customer {customer_name}.")

                                        
                             # Display updated customer table
                            st.subheader("VIP Customer Details")
                            st.write(edited_df.drop(columns=['New VIP Status']))

                            # Optionally, allow users to download the updated data
                            csv_data = convert_df(edited_df.drop(columns=['New VIP Status']))
                            excel_data = convert_df_to_excel(edited_df.drop(columns=['New VIP Status']))

                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                label="Download updated data as CSV",
                                data=csv_data,
                                file_name='updated_vip_analysis.csv',
                                mime='text/csv',
                                )
                            with col2:
                                st.download_button(
                                label="Download updated data as Excel",
                                data=excel_data,
                                file_name='updated_vip_analysis.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                )

            elif page == 'Customer Inquiry Module':
                # ------------------ Customer Inquiry Module ------------------

                st.subheader("Customer Inquiry Module")

                with st.form(key='customer_inquiry_form'):
                    st.write("Enter at least one of the following fields to search for a customer:")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        input_last_name = st.text_input("Last Name")
                    with col2:
                        input_phone_number = st.text_input("Phone Number")
                    with col3:
                        input_customer_id = st.text_input("Customer ID")

                    submit_inquiry = st.form_submit_button(label='Search')

                if submit_inquiry:
                    if not input_last_name and not input_phone_number and not input_customer_id:
                        st.error("Please enter at least one of Last Name, Phone Number, or Customer ID.")
                    else:
                        # Filter rfm_data based on inputs
                        inquiry_results = rfm_data.copy()

                        if input_last_name:
                            inquiry_results = inquiry_results[inquiry_results['Last Name'].str.contains(input_last_name, na=False)]
                        if input_phone_number:
                            inquiry_results = inquiry_results[inquiry_results['Phone Number'].astype(str).str.contains(input_phone_number)]
                        if input_customer_id:
                            inquiry_results = inquiry_results[inquiry_results['Customer ID'].astype(str).str.contains(input_customer_id)]

                        if inquiry_results.empty:
                            st.warning("No customers found matching the given criteria.")
                        else:
                            st.success(f"Found {len(inquiry_results)} customer(s) matching the criteria.")

                            # Display customer information
                            for index, customer in inquiry_results.iterrows():
                                st.markdown("---")
                                st.subheader(f"Customer ID: {customer['Customer ID']}")
                                st.write(f"**Name:** {customer['First Name']} {customer['Last Name']}")
                                st.write(f"**Phone Number:** {customer['Phone Number']}")
                                st.write(f"**VIP Status:** {customer['VIP Status']}")
                                st.write(f"**Recency:** {customer['Recency']} days")
                                st.write(f"**Frequency:** {customer['Frequency']}")
                                st.write(f"**Monetary:** {round(customer['Monetary'], 2)}")
                                st.write(f"**Segment:** {customer['RFM_segment_label']}")

                                # Fetch deal history
                                customer_deals = data[data['کد دیدار شخص معامله'] == customer['Customer ID']]
                                if customer_deals.empty:
                                    st.write("No deal history available.")
                                else:
                                    st.write("**Deal History:**")
                                    deal_history = customer_deals[['تاریخ انجام معامله', 'عنوان محصول', 'ارزش معامله', 'وضعیت معامله']].copy()
                                    # Adjust monetary values for display
                                    deal_history['ارزش معامله'] = deal_history['ارزش معامله'].round(2)
                                    st.dataframe(deal_history)

                # New Feature: Upload Excel or CSV File and Select Column Type
                st.subheader("Bulk Customer Inquiry")

                uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=['xlsx', 'csv'])
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            file_data = pd.read_csv(uploaded_file)
                        else:
                            file_data = pd.read_excel(uploaded_file)

                        st.write("File uploaded successfully!")
                        st.write("Columns in the file:", list(file_data.columns))

                        selected_column = st.selectbox("Select the column to search by", file_data.columns)

                        column_type = st.radio("What does the selected column contain?", ["Numbers", "Names", "IDs"])

                        if st.button("Search from File"):
                            if column_type == "Numbers":
                                matching_results = rfm_data[rfm_data['Phone Number'].astype(str).isin(file_data[selected_column].astype(str))]
                            elif column_type == "Names":
                                matching_results = rfm_data[rfm_data['Last Name'].isin(file_data[selected_column])]
                            elif column_type == "IDs":
                                matching_results = rfm_data[rfm_data['Customer ID'].astype(str).isin(file_data[selected_column].astype(str))]
                            else:
                                matching_results = pd.DataFrame()

                            # Separate results into existing and new users
                            file_data['Exists_in_Dataset'] = file_data[selected_column].astype(str).isin(rfm_data['Customer ID'].astype(str)) | \
                                                            file_data[selected_column].astype(str).isin(rfm_data['Phone Number'].astype(str)) | \
                                                            file_data[selected_column].isin(rfm_data['Last Name'])

                            existing_users = file_data[file_data['Exists_in_Dataset']]
                            new_users = file_data[~file_data['Exists_in_Dataset']]

                            # Display existing users
                            if not existing_users.empty:
                                st.success(f"Found {len(existing_users)} existing customer(s) from the uploaded file.")
                                st.dataframe(matching_results)

                            # Display new users (Acquisition users)
                            if not new_users.empty:
                                st.warning(f"Identified {len(new_users)} new user(s) not present in the dataset.")
                                st.subheader("Acquisition Users")
                                st.dataframe(new_users)

                    except Exception as e:
                        st.error(f"Error processing file: {e}")


        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload an Excel file to proceed.")

if __name__ == '__main__':
    main()
