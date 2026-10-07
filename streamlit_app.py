import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CLEANED_CSV = BASE_DIR / "cleaned_survey_data.csv"
CLUSTERED_CSV = BASE_DIR / "KMeans_clustered_data.csv"

FEATURES = [
    'Age_Num', 'Spend_Num', 'MoMo_Num', 'Call_Num', 'SMS_Num',
    'Tenure_Num', 'Recharge_Num', 'Satisfaction', 'RFM_Recency',
    'RFM_Frequency', 'RFM_Monetary', 'Engagement_Score', 'Loyalty_Score'
]

PLOT_FEATURES = [
    'RFM_Monetary',
    'Recharge_Num',
    'RFM_Frequency',
    'Engagement_Score',
    'Loyalty_Score',
    'Spend_Num',
    'Satisfaction',
    'Tenure_Num',
]

PLOT_FEATURE_LABELS = {
    'RFM_Monetary': 'Monetary value',
    'Recharge_Num': 'Recharge frequency',
    'RFM_Frequency': 'Usage frequency',
    'Engagement_Score': 'Engagement score',
    'Loyalty_Score': 'Loyalty score',
    'Spend_Num': 'Monthly spend level',
    'Satisfaction': 'Satisfaction',
    'Tenure_Num': 'Customer tenure',
}

SEGMENT_NAMES = {
    0: 'High-Spend Infrequent Rechargers',
    1: 'Low-Spend Daily Rechargers',
    2: 'Budget Passive Users',
    3: 'High-Value Heavy Users'
}

AGE_ORDER = {
    'Under 18': 0,
    '18–24': 1,
    '25–34': 2,
    '35–44': 3,
    '45–54': 4,
    '55+': 5,
}

SPEND_ORDER = {
    'Less than TZS 5,000': 1,
    '5,000–10,000': 2,
    '10,001–20,000': 3,
    '20,001–50,000': 4,
    'More than 50,000': 5,
}

MOMO_ORDER = {
    'None': 0,
    '1–10': 1,
    '11–30': 2,
    '31–60': 3,
    'More than 60': 4,
}

CALL_ORDER = {
    'Less than 30': 1,
    '30–60': 2,
    '61–120': 3,
    '121–300': 4,
    'More than 300': 5,
}

SMS_ORDER = {
    'None': 0,
    '1–10': 1,
    '11–30': 2,
    'More than 30': 3,
}

TENURE_ORDER = {
    'Less than 1 year': 1,
    '1–2 years': 2,
    '3–5 years': 3,
    'More than 5 years': 4,
}

RECHARGE_ORDER = {
    'Occasionally': 1,
    'Monthly': 2,
    'Weekly': 3,
    'Daily': 4,
}

RAW_COLUMNS = [
    'Age', 'Gender', 'District', 'Occupation', 'Network', 'Years_Used',
    'Monthly_Spend', 'MoMo_Transactions', 'Call_Minutes_Week', 'SMS_Week',
    'Primary_Service', 'Recharge_Frequency', 'Changed_Network', 'Reason_Changed',
    'Satisfaction', 'Heard_AI', 'AI_for_Telecom', 'Agree_Personalized'
]

RAW_HEADER_MAP = {
    'age': 'Age',
    'gender': 'Gender',
    'district of residence': 'District',
    'occupation': 'Occupation',
    'which mobile network do you mainly use?': 'Network',
    'how many years have you used this network?': 'Years_Used',
    'approximately how much airtime do you spend per month?/ approximately how much do you spend on internet bundles per month?': 'Monthly_Spend',
    'how many mobile money transactions do you make in a month?': 'MoMo_Transactions',
    'approximately how many minutes do you spend on phone calls each week?': 'Call_Minutes_Week',
    'how many sms messages do you send each week?': 'SMS_Week',
    'which service do you use most?': 'Primary_Service',
    'how often do you recharge airtime?': 'Recharge_Frequency',
    'have you ever changed your mobile network?': 'Changed_Network',
    'if yes, why?': 'Reason_Changed',
    'overall satisfaction with your current network': 'Satisfaction',
    'have you heard about artificial intelligence or machine learning?': 'Heard_AI',
    'do you think telecom companies should use ai to improve customer services?': 'AI_for_Telecom',
    'would you agree to receive personalized offers based on your usage behavior?': 'Agree_Personalized',
}


def normalize_text(value: str) -> str:
    return ' '.join(str(value).strip().lower().replace('\n', ' ').split())


def map_raw_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        norm = normalize_text(col)
        if norm in RAW_HEADER_MAP:
            rename[col] = RAW_HEADER_MAP[norm]
    if rename:
        df = df.rename(columns=rename)
    return df


def load_cleaned_data() -> pd.DataFrame:
    if not CLEANED_CSV.exists():
        st.error(f"Missing cleaned dataset: {CLEANED_CSV}")
        st.stop()
    return pd.read_csv(CLEANED_CSV)


def load_clustered_data() -> pd.DataFrame:
    if not CLUSTERED_CSV.exists():
        st.error(f"Missing clustered dataset: {CLUSTERED_CSV}")
        st.stop()
    return pd.read_csv(CLUSTERED_CSV)


def build_scaler_and_model(df: pd.DataFrame):
    scaler = MinMaxScaler()
    X = df[FEATURES]
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
    model.fit(X_scaled)
    return scaler, model


def add_cluster_ellipse(ax, x_values, y_values, color):
    if len(x_values) < 3:
        return

    covariance = np.cov(x_values, y_values)
    if not np.all(np.isfinite(covariance)):
        return

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        return

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.4 * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        (np.mean(x_values), np.mean(y_values)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        linewidth=2,
        zorder=1,
    )
    ax.add_patch(ellipse)


def normalize_raw_strings(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    text_columns = [
        'Age', 'Gender', 'District', 'Occupation', 'Network', 'Years_Used',
        'Monthly_Spend', 'MoMo_Transactions', 'Call_Minutes_Week', 'SMS_Week',
        'Primary_Service', 'Recharge_Frequency', 'Changed_Network', 'Reason_Changed',
        'Heard_AI', 'AI_for_Telecom', 'Agree_Personalized'
    ]
    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().replace({'nan': pd.NA, '': pd.NA})
    return data


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = normalize_raw_strings(data)

    if 'Satisfaction' in data.columns:
        data['Satisfaction'] = pd.to_numeric(data['Satisfaction'], errors='coerce')

    data['MoMo_Transactions'] = data['MoMo_Transactions'].fillna('None')
    data['SMS_Week'] = data['SMS_Week'].fillna('None')
    data['Satisfaction'] = data['Satisfaction'].fillna(
        data['Satisfaction'].median() if data['Satisfaction'].notna().any() else 3
    )
    data['Years_Used'] = data['Years_Used'].fillna(
        data['Years_Used'].mode().iloc[0]
        if data['Years_Used'].notna().any()
        else 'Less than 1 year'
    )
    data['AI_for_Telecom'] = data['AI_for_Telecom'].fillna(
        data['AI_for_Telecom'].mode().iloc[0]
        if data['AI_for_Telecom'].notna().any()
        else 'Yes'
    )
    data['Agree_Personalized'] = data['Agree_Personalized'].fillna(
        data['Agree_Personalized'].mode().iloc[0]
        if data['Agree_Personalized'].notna().any()
        else 'Agree'
    )
    data['Reason_Changed'] = data['Reason_Changed'].fillna('N/A')
    data['Changed_Network'] = data['Changed_Network'].fillna('No')
    return data


def preprocess_input(input_data: dict) -> pd.DataFrame:
    data = pd.DataFrame([input_data])
    data = fill_missing_values(data)

    data['Age_Num'] = data['Age'].map(AGE_ORDER)
    data['Spend_Num'] = data['Monthly_Spend'].map(SPEND_ORDER)
    data['MoMo_Num'] = data['MoMo_Transactions'].map(MOMO_ORDER)
    data['Call_Num'] = data['Call_Minutes_Week'].map(CALL_ORDER)
    data['SMS_Num'] = data['SMS_Week'].map(SMS_ORDER)
    data['Tenure_Num'] = data['Years_Used'].map(TENURE_ORDER)
    data['Recharge_Num'] = data['Recharge_Frequency'].map(RECHARGE_ORDER)

    data['RFM_Recency'] = data['Recharge_Num']
    data['RFM_Frequency'] = (
        data['Call_Num'] + data['MoMo_Num'] + data['SMS_Num']
    ) / 3
    data['RFM_Monetary'] = data['Spend_Num']

    data['Engagement_Score'] = (
        data['Spend_Num'] * 0.35
        + data['Call_Num'] * 0.25
        + data['MoMo_Num'] * 0.20
        + data['Recharge_Num'] * 0.10
        + data['SMS_Num'] * 0.10
    )

    data['Loyalty_Score'] = data['Tenure_Num'] * 0.6 + data['Satisfaction'] * 0.4

    return data[FEATURES]


def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = df[RAW_COLUMNS].copy()
    data = fill_missing_values(data)

    data['Age_Num'] = data['Age'].map(AGE_ORDER)
    data['Spend_Num'] = data['Monthly_Spend'].map(SPEND_ORDER)
    data['MoMo_Num'] = data['MoMo_Transactions'].map(MOMO_ORDER)
    data['Call_Num'] = data['Call_Minutes_Week'].map(CALL_ORDER)
    data['SMS_Num'] = data['SMS_Week'].map(SMS_ORDER)
    data['Tenure_Num'] = data['Years_Used'].map(TENURE_ORDER)
    data['Recharge_Num'] = data['Recharge_Frequency'].map(RECHARGE_ORDER)

    data['RFM_Recency'] = data['Recharge_Num']
    data['RFM_Frequency'] = (
        data['Call_Num'] + data['MoMo_Num'] + data['SMS_Num']
    ) / 3
    data['RFM_Monetary'] = data['Spend_Num']

    data['Engagement_Score'] = (
        data['Spend_Num'] * 0.35
        + data['Call_Num'] * 0.25
        + data['MoMo_Num'] * 0.20
        + data['Recharge_Num'] * 0.10
        + data['SMS_Num'] * 0.10
    )

    data['Loyalty_Score'] = data['Tenure_Num'] * 0.6 + data['Satisfaction'] * 0.4

    return data[FEATURES]


def make_prediction_result(input_data: dict, cluster: int, segment: str) -> pd.DataFrame:
    raw = pd.DataFrame([input_data])
    raw['KMeans_Cluster'] = cluster
    raw['Segment_Label'] = segment
    raw['Age_Num'] = raw['Age'].map(AGE_ORDER)
    raw['Spend_Num'] = raw['Monthly_Spend'].map(SPEND_ORDER)
    raw['MoMo_Num'] = raw['MoMo_Transactions'].map(MOMO_ORDER)
    raw['Call_Num'] = raw['Call_Minutes_Week'].map(CALL_ORDER)
    raw['SMS_Num'] = raw['SMS_Week'].map(SMS_ORDER)
    raw['Tenure_Num'] = raw['Years_Used'].map(TENURE_ORDER)
    raw['Recharge_Num'] = raw['Recharge_Frequency'].map(RECHARGE_ORDER)
    raw['RFM_Recency'] = raw['Recharge_Num']
    raw['RFM_Frequency'] = (
        raw['Call_Num'] + raw['MoMo_Num'] + raw['SMS_Num']
    ) / 3
    raw['RFM_Monetary'] = raw['Spend_Num']
    raw['Engagement_Score'] = (
        raw['Spend_Num'] * 0.35
        + raw['Call_Num'] * 0.25
        + raw['MoMo_Num'] * 0.20
        + raw['Recharge_Num'] * 0.10
        + raw['SMS_Num'] * 0.10
    )
    raw['Loyalty_Score'] = raw['Tenure_Num'] * 0.6 + raw['Satisfaction'] * 0.4
    return raw


def build_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='ClusterPrediction')
    return buffer.getvalue()


MAX_UPLOAD_MB = 512


def main():
    st.set_page_config(page_title='Telecom KMeans Cluster Prediction', layout='centered')
    st.title('Telecom Customer Cluster Predictor')
    st.write(
        'Enter a single respondent profile below, or upload raw dirty survey data. '
        'The app will clean the data, predict KMeans clusters, and assign segment labels.'
    )
   
    # Small visual improvements and sidebar settings
    st.markdown(
        """
        <style>
        .stApp { max-width: 1100px; margin: 0 auto; padding: 1rem; }
        .stButton>button { background-color: #0b6efd; color: white; }
        .card { background: #FFFFFF; padding: 0.8rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(197, 79, 79, 0.08); }
        .dark .card { background:#ffffff;; color: #d1d5db; }
         h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
          /* Make selectbox label black */
    div[data-testid="stSelectbox"] label {
        color: black !important;
        font-weight: 1000;
    }

    /* Make selected value black */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color:yellow !important;
    }

    /* Make dropdown options black */
    div[data-testid="stSelectbox"] * {
        color: yellow !important;
    }
    .stMarkdown p,
[data-testid="stMarkdownContainer"] p {
    color: black !important;
}

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.title('Settings')
    preview_rows = st.sidebar.slider('Preview rows', min_value=5, max_value=200, value=10)
    compact_form = st.sidebar.checkbox('Compact respondent form', value=False)
    bright_theme = st.sidebar.checkbox('Bright theme', value=True)

    if bright_theme:
        st.markdown(
            """
            <style>
            body {
                background-color: #f8fafc;
                color: #0f172a;
            }
            .stApp {
                background-color: #ffffff;
            }
            .stSidebar {
                background-color: #f1f5f9;
            }
            .stButton>button,
            .stDownloadButton>button {
                background-color: #0b6efd;
                color: #ffffff;
                border: none;
            }
            .stTextInput>div>div>input,
            .stSelectbox>div>div,
            .stSlider>div>div {
                color: #0f172a;
            }
            .stMarkdown p,
            [data-testid="stMarkdownContainer"] p,
            .stTextInput label,
            .stSelectbox label,
            .stSlider label {
                color: #0f172a !important;
            }
            .streamlit-expanderHeader,
            .stExpanderHeader,
            [data-testid="stExpander"] > div {
                background-color: #e2e8f0 !important;
                color: #0f172a !important;
                border: 1px solid #cbd5e1 !important;
                border-radius: 0.75rem !important;
                padding: 0.7rem 1rem !important;
            }
            .streamlit-expanderHeader:hover,
            .stExpanderHeader:hover,
            [data-testid="stExpander"] > div:hover {
                background-color: #dbeafe !important;
            }
            .stExpanderContent,
            .streamlit-expanderContent,
            [data-testid="stExpander"] section {
                background-color: #f8fafc !important;
                padding: 1rem !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 0 0 0.75rem 0.75rem !important;
                margin-top: -1px !important;
            }
            [data-testid="stExpander"] .stMarkdown p,
            [data-testid="stExpander"] label,
            [data-testid="stExpander"] select,
            [data-testid="stExpander"] input {
                color: #0f172a !important;
            }
            .stTooltip {
                background-color: #f8fafc;
                color: #0f172a;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    df_cleaned = load_cleaned_data()
    df_clustered = load_clustered_data()
    scaler, model = build_scaler_and_model(df_cleaned)

    st.subheader('Respondent Profile')
    if compact_form:
        with st.form('input_form'):
            col1, col2 = st.columns(2)
            with col1:
                with st.expander('1. Personal information', expanded=True):
                    age = st.selectbox('Age', ['Under 18', '18–24', '25–34', '35–44', '45–54', '55+'])
                    gender = st.selectbox('Gender', ['Male', 'Female', 'Prefer not to say'])
                    district = st.selectbox('District', ['Ilala', 'Kigamboni', 'Kinondoni', 'Temeke', 'Ubungo'])
                    occupation = st.selectbox('Occupation', ['Business owner', 'Employed', 'Self-employed', 'Student', 'Unemployed'])

                with st.expander('2. Customer usage / tenure', expanded=True):
                    years_used = st.selectbox('Years Used', ['Less than 1 year', '1–2 years', '3–5 years', 'More than 5 years'])
                    call_minutes = st.selectbox('Call Minutes per Week', ['Less than 30', '30–60', '61–120', '121–300', 'More than 300'])
                    sms_week = st.selectbox('SMS per Week', ['None', '1–10', '11–30', 'More than 30'])
                    recharge_frequency = st.selectbox('Recharge Frequency', ['Occasionally', 'Monthly', 'Weekly', 'Daily'])

                with st.expander('3. Spending behavior', expanded=True):
                    monthly_spend = st.selectbox('Monthly Spend', ['Less than TZS 5,000', '5,000–10,000', '10,001–20,000', '20,001–50,000', 'More than 50,000'])
                    momo_transactions = st.selectbox('MoMo Transactions', ['None', '1–10', '11–30', '31–60', 'More than 60'])

            with col2:
                with st.expander('4. Network & service information', expanded=True):
                    network = st.selectbox('Network', ['Airtel', 'Halotel', 'TTCL', 'Tigo', 'Vodacom'])
                    primary_service = st.selectbox('Primary Service', ['Internet', 'Mixed Usage', 'Mobile Money', 'SMS', 'Voice Calls'])

                with st.expander('5. Customer switching behavior', expanded=True):
                    changed_network = st.selectbox('Changed Network', ['Yes', 'No'])
                    if changed_network == 'Yes':
                        reason_changed = st.selectbox('Reason Changed', ['Better customer service', 'Better internet', 'Better network coverage', 'Lower prices', 'Promotions', 'Other'])
                    else:
                        reason_changed = 'N/A'

                with st.expander('6. Customer satisfaction', expanded=True):
                    satisfaction = st.slider('Satisfaction (1-5)', min_value=1, max_value=5, value=3)

                with st.expander('7. AI awareness & adoption', expanded=True):
                    heard_ai = st.selectbox('Heard about AI', ['Yes', 'No'])
                    ai_for_telecom = st.selectbox('Would use AI for Telecom', ['Yes', 'No'])

                with st.expander('8. Customer personalization / marketing preference', expanded=True):
                    agree_personalized = st.selectbox('Agree with Personalized Offers', ['Strongly Agree', 'Agree', 'Neutral', 'Strongly Disagree'])

            submitted = st.form_submit_button('Predict Cluster')
    else:
        with st.form('input_form'):
            col1, col2 = st.columns(2)
            with col1:
                with st.expander('1. Personal information', expanded=True):
                    age = st.selectbox('Age', ['Under 18', '18–24', '25–34', '35–44', '45–54', '55+'])
                    gender = st.selectbox('Gender', ['Male', 'Female', 'Prefer not to say'])
                    district = st.selectbox('District', ['Ilala', 'Kigamboni', 'Kinondoni', 'Temeke', 'Ubungo'])
                    occupation = st.selectbox('Occupation', ['Business owner', 'Employed', 'Self-employed', 'Student', 'Unemployed'])

                with st.expander('2. Customer usage / tenure', expanded=True):
                    years_used = st.selectbox('Years Used', ['Less than 1 year', '1–2 years', '3–5 years', 'More than 5 years'])
                    call_minutes = st.selectbox('Call Minutes per Week', ['Less than 30', '30–60', '61–120', '121–300', 'More than 300'])
                    sms_week = st.selectbox('SMS per Week', ['None', '1–10', '11–30', 'More than 30'])
                    recharge_frequency = st.selectbox('Recharge Frequency', ['Occasionally', 'Monthly', 'Weekly', 'Daily'])

                with st.expander('3. Spending behavior', expanded=True):
                    monthly_spend = st.selectbox('Monthly Spend', ['Less than TZS 5,000', '5,000–10,000', '10,001–20,000', '20,001–50,000', 'More than 50,000'])
                    momo_transactions = st.selectbox('MoMo Transactions', ['None', '1–10', '11–30', '31–60', 'More than 60'])

            with col2:
                with st.expander('4. Network & service information', expanded=True):
                    network = st.selectbox('Network', ['Airtel', 'Halotel', 'TTCL', 'Tigo', 'Vodacom'])
                    primary_service = st.selectbox('Primary Service', ['Internet', 'Mixed Usage', 'Mobile Money', 'SMS', 'Voice Calls'])

                with st.expander('5. Customer switching behavior', expanded=True):
                    changed_network = st.selectbox('Changed Network', ['Yes', 'No'])
                    if changed_network == 'Yes':
                        reason_changed = st.selectbox('Reason Changed', ['Better customer service', 'Better internet', 'Better network coverage', 'Lower prices', 'Promotions', 'Other'])
                    else:
                        reason_changed = 'N/A'

                with st.expander('6. Customer satisfaction', expanded=True):
                    satisfaction = st.slider('Satisfaction (1-5)', min_value=1, max_value=5, value=3)

                with st.expander('7. AI awareness & adoption', expanded=True):
                    heard_ai = st.selectbox('Heard about AI', ['Yes', 'No'])
                    ai_for_telecom = st.selectbox('Would use AI for Telecom', ['Yes', 'No'])

                with st.expander('8. Customer personalization / marketing preference', expanded=True):
                    agree_personalized = st.selectbox('Agree with Personalized Offers', ['Strongly Agree', 'Agree', 'Neutral', 'Strongly Disagree'])

            submitted = st.form_submit_button('Predict Cluster')

    if submitted:
        input_data = {
            'Age': age,
            'Gender': gender,
            'District': district,
            'Occupation': occupation,
            'Network': network,
            'Years_Used': years_used,
            'Monthly_Spend': monthly_spend,
            'MoMo_Transactions': momo_transactions,
            'Call_Minutes_Week': call_minutes,
            'SMS_Week': sms_week,
            'Primary_Service': primary_service,
            'Recharge_Frequency': recharge_frequency,
            'Changed_Network': changed_network,
            'Reason_Changed': reason_changed,
            'Satisfaction': float(satisfaction),
            'Heard_AI': heard_ai,
            'AI_for_Telecom': ai_for_telecom,
            'Agree_Personalized': agree_personalized,
        }

        features_df = preprocess_input(input_data)
        scaled_input = scaler.transform(features_df)
        cluster = int(model.predict(scaled_input)[0])
        segment = SEGMENT_NAMES.get(cluster, 'Unknown Segment')

        st.success(f'Result: cluster {cluster} — {segment}')
        st.write('### Raw input with cluster assignment')

        result_df = make_prediction_result(input_data, cluster, segment)
        st.dataframe(result_df.T)

        excel_bytes = build_excel_bytes(result_df)
        output_path = BASE_DIR / 'individual_cluster_prediction.xlsx'
        output_path.write_bytes(excel_bytes)

        st.download_button(
            label='Download Excel file',
            data=excel_bytes,
            file_name='individual_cluster_prediction.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

        st.info(f'Raw Excel file saved to: {output_path}')

    st.write('---')
    st.subheader('Batch Excel Upload')
    st.write(
        'Upload a raw dirty survey file (`.xlsx`, `.xls`, or `.csv`) with the original survey columns. '
        f'Maximum file size is {MAX_UPLOAD_MB} MB. '
        'The app will clean missing values and append cluster / segment labels.'
    )

    uploaded_file = st.file_uploader('Upload raw survey file', type=['xlsx', 'xls', 'csv'])
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / 1024**2
        if file_size_mb > MAX_UPLOAD_MB:
            st.error(f'File is too large ({file_size_mb:.1f} MB). Please upload a file smaller than {MAX_UPLOAD_MB} MB.')
        else:
            try:
                with st.spinner('Processing batch file — this may take a moment...'):
                    if uploaded_file.name.lower().endswith('.csv'):
                        batch_df = pd.read_csv(uploaded_file)
                    else:
                        batch_df = pd.read_excel(uploaded_file)
                    batch_df = map_raw_headers(batch_df)
                    batch_features = preprocess_batch(batch_df)
                    batch_scaled = scaler.transform(batch_features)
                    batch_clusters = model.predict(batch_scaled)
                    batch_df['KMeans_Cluster'] = batch_clusters
                    batch_df['Segment_Label'] = [SEGMENT_NAMES.get(int(c), 'Unknown Segment') for c in batch_clusters]

                st.success(f'Batch file processed: {len(batch_df)} rows.')

                # Key metrics and quick visualization
                m1, m2, m3 = st.columns([1, 1, 1])
                m1.metric('Rows', len(batch_df))
                m2.metric('Unique Clusters', int(batch_df['KMeans_Cluster'].nunique()))
                top_segment = batch_df['Segment_Label'].mode().iat[0] if not batch_df['Segment_Label'].empty else 'N/A'
                m3.metric('Top Segment', top_segment)

                st.markdown('**Cluster distribution**')
                st.bar_chart(batch_df['Segment_Label'].value_counts())

                # Scatter plot: use saved KMeans output so the visual shows the real fitted clusters.
                st.markdown('**Real KMeans clusters**')
                try:
                    plot_columns = FEATURES + ['KMeans_Cluster', 'Segment_Label']
                    plot_df = df_clustered[plot_columns].copy()

                    scatter_mode = st.radio(
                        'Scatter view',
                        ['Cluster profile map', 'Cluster separation map', 'PCA cluster map', 'Selected feature pair'],
                        horizontal=True,
                    )
                    if scatter_mode == 'Cluster profile map':
                        scatter_x_col, scatter_y_col = st.columns(2)
                        with scatter_x_col:
                            x_feature = st.selectbox(
                                'Business X-axis',
                                PLOT_FEATURES,
                                index=PLOT_FEATURES.index('RFM_Monetary'),
                                format_func=lambda feature: PLOT_FEATURE_LABELS.get(feature, feature),
                            )
                        with scatter_y_col:
                            y_feature = st.selectbox(
                                'Business Y-axis',
                                PLOT_FEATURES,
                                index=PLOT_FEATURES.index('Recharge_Num'),
                                format_func=lambda feature: PLOT_FEATURE_LABELS.get(feature, feature),
                            )

                        plot_df[x_feature] = pd.to_numeric(plot_df[x_feature], errors='coerce')
                        plot_df[y_feature] = pd.to_numeric(plot_df[y_feature], errors='coerce')
                        plot_df['KMeans_Cluster'] = pd.to_numeric(plot_df['KMeans_Cluster'], errors='coerce')
                        plot_df = plot_df.dropna(subset=[x_feature, y_feature, 'KMeans_Cluster'])

                        rng = np.random.default_rng(42)
                        x_range = max(plot_df[x_feature].max() - plot_df[x_feature].min(), 1)
                        y_range = max(plot_df[y_feature].max() - plot_df[y_feature].min(), 1)
                        plot_df['Scatter_X'] = plot_df[x_feature] + rng.normal(0, x_range * 0.025, len(plot_df))
                        plot_df['Scatter_Y'] = plot_df[y_feature] + rng.normal(0, y_range * 0.025, len(plot_df))
                        x_label = PLOT_FEATURE_LABELS.get(x_feature, x_feature)
                        y_label = PLOT_FEATURE_LABELS.get(y_feature, y_feature)
                        title = f'Cluster profile map: {x_label} vs {y_label}'
                    elif scatter_mode == 'Selected feature pair':
                        scatter_x_col, scatter_y_col = st.columns(2)
                        with scatter_x_col:
                            x_feature = st.selectbox(
                                'X-axis feature',
                                FEATURES,
                                index=FEATURES.index('Engagement_Score'),
                            )
                        with scatter_y_col:
                            y_feature = st.selectbox(
                                'Y-axis feature',
                                FEATURES,
                                index=FEATURES.index('Loyalty_Score'),
                            )

                        plot_df[x_feature] = pd.to_numeric(plot_df[x_feature], errors='coerce')
                        plot_df[y_feature] = pd.to_numeric(plot_df[y_feature], errors='coerce')
                        plot_df = plot_df.dropna(subset=[x_feature, y_feature, 'KMeans_Cluster'])
                        plot_df['Scatter_X'] = plot_df[x_feature]
                        plot_df['Scatter_Y'] = plot_df[y_feature]
                        x_label = x_feature
                        y_label = y_feature
                        title = f'Real KMeans clusters: {x_feature} vs {y_feature}'
                    elif scatter_mode == 'Cluster separation map':
                        for feature in FEATURES:
                            plot_df[feature] = pd.to_numeric(plot_df[feature], errors='coerce')
                        plot_df['KMeans_Cluster'] = pd.to_numeric(plot_df['KMeans_Cluster'], errors='coerce')
                        plot_df = plot_df.dropna(subset=FEATURES + ['KMeans_Cluster'])

                        X_scaled = scaler.transform(plot_df[FEATURES])
                        y_clusters = plot_df['KMeans_Cluster'].astype(int)
                        lda = LinearDiscriminantAnalysis(n_components=2)
                        separation_values = lda.fit_transform(X_scaled, y_clusters)
                        plot_df['Scatter_X'] = separation_values[:, 0]
                        plot_df['Scatter_Y'] = separation_values[:, 1]
                        x_label = 'Cluster axis 1'
                        y_label = 'Cluster axis 2'
                        title = 'Real KMeans clusters: separation map'
                    else:
                        for feature in FEATURES:
                            plot_df[feature] = pd.to_numeric(plot_df[feature], errors='coerce')
                        plot_df['KMeans_Cluster'] = pd.to_numeric(plot_df['KMeans_Cluster'], errors='coerce')
                        plot_df = plot_df.dropna(subset=FEATURES + ['KMeans_Cluster'])
                        pca = PCA(n_components=2, random_state=42)
                        pca_values = pca.fit_transform(scaler.transform(plot_df[FEATURES]))
                        plot_df['Scatter_X'] = pca_values[:, 0]
                        plot_df['Scatter_Y'] = pca_values[:, 1]
                        x_label = 'PC1'
                        y_label = 'PC2'
                        title = 'Real KMeans clusters: PCA projection of all features'

                    plot_df['KMeans_Cluster'] = pd.to_numeric(plot_df['KMeans_Cluster'], errors='coerce')

                    if plot_df.empty:
                        st.info('Scatter plot unavailable (no numeric feature values in the clustered dataset).')
                    else:
                        max_points = 5000
                        if len(plot_df) > max_points:
                            plot_df = plot_df.sample(n=max_points, random_state=1)
                            st.warning(f'Showing random sample of {max_points} points for performance.')

                        fig, ax = plt.subplots(figsize=(6, 4))
                        cmap = plt.get_cmap('tab10')
                        for cluster_id in sorted(plot_df['KMeans_Cluster'].astype(int).unique()):
                            cluster_points = plot_df[plot_df['KMeans_Cluster'].astype(int) == cluster_id]
                            color = cmap(cluster_id % 10)
                            segment_label = SEGMENT_NAMES.get(cluster_id, f'Cluster {cluster_id}')
                            ax.scatter(
                                cluster_points['Scatter_X'],
                                cluster_points['Scatter_Y'],
                                color=color,
                                alpha=0.65,
                                s=24,
                                label=f'C{cluster_id}: {segment_label}',
                                zorder=2,
                            )
                            if scatter_mode == 'Cluster separation map':
                                add_cluster_ellipse(
                                    ax,
                                    cluster_points['Scatter_X'].to_numpy(),
                                    cluster_points['Scatter_Y'].to_numpy(),
                                    color,
                                )
                                centroid_x = cluster_points['Scatter_X'].mean()
                                centroid_y = cluster_points['Scatter_Y'].mean()
                                ax.scatter(
                                    centroid_x,
                                    centroid_y,
                                    color=color,
                                    edgecolor='black',
                                    linewidth=1.2,
                                    s=160,
                                    marker='X',
                                    zorder=3,
                                )
                                ax.text(
                                    centroid_x,
                                    centroid_y,
                                    f' C{cluster_id}',
                                    fontsize=9,
                                    fontweight='bold',
                                    va='center',
                                    zorder=4,
                                )
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(title)
                        ax.legend(title='Cluster', fontsize=7, loc='best')
                        ax.grid(alpha=0.2)

                        if scatter_mode == 'Cluster profile map':
                            profile = (
                                plot_df.groupby('KMeans_Cluster')
                                .agg(
                                    Segment=('Segment_Label', 'first'),
                                    N=('KMeans_Cluster', 'size'),
                                    Mean_X=(x_feature, 'mean'),
                                    Mean_Y=(y_feature, 'mean'),
                                )
                                .round(2)
                            )
                            for cluster_id, row in profile.iterrows():
                                ax.scatter(
                                    row['Mean_X'],
                                    row['Mean_Y'],
                                    color=cmap(int(cluster_id) % 10),
                                    edgecolor='black',
                                    linewidth=1.4,
                                    s=220,
                                    marker='X',
                                    zorder=5,
                                )
                                ax.text(
                                    row['Mean_X'],
                                    row['Mean_Y'],
                                    f' C{int(cluster_id)}',
                                    fontsize=10,
                                    fontweight='bold',
                                    va='center',
                                    zorder=6,
                                )
                            st.pyplot(fig)
                            st.caption('Small jitter is applied only to separate overlapping survey-score points; cluster centroids use the real mean values.')
                            st.dataframe(
                                profile.rename(columns={'Mean_X': f'Mean {x_label}', 'Mean_Y': f'Mean {y_label}'})
                            )
                        else:
                            st.pyplot(fig)

                except Exception as exc:
                    st.info('Scatter plot unavailable (error plotting engineered features).')
                    st.exception(exc)

                # Optional PCA of scaled model input space
                if st.sidebar.checkbox('Show PCA projection of feature space', value=False):
                    try:
                        st.markdown('**PCA projection (2 components) of scaled features**')
                        X_scaled = scaler.transform(batch_features)
                        pca = PCA(n_components=2, random_state=1)
                        proj = pca.fit_transform(X_scaled)
                        proj_df = pd.DataFrame(proj, columns=['PC1', 'PC2'])
                        proj_df['KMeans_Cluster'] = batch_df['KMeans_Cluster'].astype(int).values

                        max_points = 5000
                        if len(proj_df) > max_points:
                            proj_df = proj_df.sample(n=max_points, random_state=1)
                            st.warning(f'Showing random sample of {max_points} points for PCA.')

                        fig, ax = plt.subplots(figsize=(6, 4))
                        scatter = ax.scatter(proj_df['PC1'], proj_df['PC2'], c=proj_df['KMeans_Cluster'], cmap='tab10', alpha=0.7)
                        ax.set_xlabel('PC1')
                        ax.set_ylabel('PC2')
                        ax.set_title('PCA of scaled features')
                        legend1 = ax.legend(*scatter.legend_elements(), title='Cluster')
                        ax.add_artist(legend1)
                        st.pyplot(fig)
                    except Exception as exc:
                        st.info('PCA projection unavailable.')
                        st.exception(exc)

                st.write(f'Showing preview of up to {preview_rows} rows:')
                st.dataframe(batch_df.head(preview_rows))

                with st.expander('Show full batch results'):
                    if len(batch_df) > 2000:
                        st.warning('Rendering a large table may take some time in the browser.')
                    st.dataframe(batch_df)

                out_bytes = build_excel_bytes(batch_df)
                st.download_button(
                    label='Download batch cluster result',
                    data=out_bytes,
                    file_name='batch_cluster_predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error('Unable to process file. Please check that the file uses the expected survey columns and is not corrupted.')
                st.exception(exc)


if __name__ == '__main__':
    main()
