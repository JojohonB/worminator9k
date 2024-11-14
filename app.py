import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Worminator", layout="wide")

MODEL_PATH = 'model/model.joblib'
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

#CSS
st.markdown("""
    <style>
    .stApp {
        background-color: var(--background-color);
    }
    .title {
        font-size: 3em;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.1em;
        color: var(--text-color);
    }
    .subtitle {
        font-size: 1.5em;
        text-align: center;
        margin-top: 0;
        margin-bottom: 0.5em;
        color: var(--text-color);
    }
    /* Text Elements */
    p, label, h1, h2, h3 {
        color: var(--text-color);
    }
    /* Button Styling */
    div.stButton > button:first-child {
        background-color: #e74c3c;
        color: white; /* Text color */
        height: 50px;
        width: 200px;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
        margin: 20px auto;  /* Center horizontally */
        display: block;      /* Required for margin auto to work */
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    /* Hover state */
    div.stButton > button:first-child:hover {
        background-color: #c0392b;
        color: white; /* Ensure text remains white */
    }

    /* Active state (when the button is clicked) */
    div.stButton > button:first-child:active {
        background-color: #b03a2e;
        color: white; /* Ensure text remains white */
    }

    /* Focus state (when the button is focused) */
    div.stButton > button:first-child:focus {
        background-color: #c0392b;
        color: white; /* Ensure text remains white */
        outline: none; /* Remove default focus outline */
    }

    /* Ensure the button text color remains white in all other states */
    div.stButton > button:first-child:visited {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">Worminator</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Gender Prediction Model </p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.9em; color: #666666;'>version beta 0.1, made by jojohonB</p>", unsafe_allow_html=True)
st.write("Enter your measurements (in centimeters):")

torso_measurements = [
    ("Stature", "stature", "Your full height."),
    ("Biacromial Breadth", "biacromialbreadth", "Width across your shoulders."),
    ("Chest Circumference", "chestcircumference", "Circumference around your chest."),
    ("Buttock Circumference", "buttockcircumference", "Circumference around your buttocks."),
    ("Waist Circumference", "waistcircumference", "Circumference around your waist."),
    ("Hip Breadth", "hipbreadth", "Width across your hips."),
]

limb_measurements = [
    ("Forearm Circumference Flexed", "forearmcircumferenceflexed", "Circumference of your forearm when flexed."),
    ("Wrist Circumference", "wristcircumference", "Circumference around your wrist."),
    ("Calf Circumference", "calfcircumference", "Circumference around your calf."),
    ("Ankle Circumference", "anklecircumference", "Circumference around your ankle."),
    ("Foot Length", "footlength", "Length of your foot."),
    ("Forearm-Hand Length", "forearmhandlength", "Length from your elbow to the tip of your fingers."),
]

col_torso, col_limb = st.columns(2)

measurements = {}

with col_torso:
    st.header("Torso Measurements")
    for label, var_name, help_text in torso_measurements:
        measurements[var_name] = st.number_input(
            label, min_value=0.0, format="%.2f", help=help_text, key=var_name
        )

with col_limb:
    st.header("Limb Measurements")
    for label, var_name, help_text in limb_measurements:
        measurements[var_name] = st.number_input(
            label, min_value=0.0, format="%.2f", help=help_text, key=var_name
        )

ratio_descriptions = {
    'WHR': 'waist to hip ratio',
    'HBS': 'hip breadth to stature ratio',
    'CS': 'chest circumference to stature ratio',
    'FSR': 'forearm circumference flexed to stature ratio',
    'CBR': 'calf circumference to buttock circumference ratio',
    'BBSR': 'biacromial breadth to stature ratio',
    'BBHB': 'biacromial breadth to hip breadth ratio',
    'ANKLS': 'ankle circumference to stature ratio',
    'FLS': 'forearm-hand length to stature ratio',
    'WCS': 'wrist circumference to stature ratio'
}

# Interpretation
interpretation_templates = {
    'WHR': {
        'male': "Your waist to hip ratio is higher than average for females, indicating a less pronounced waist relative to hips, leaning towards male.",
        'female': "Your waist to hip ratio is lower than average for males, indicating a more pronounced waist relative to hips, leaning towards female."
    },
    'HBS': {
        'male': "Your hip breadth to stature ratio is low, suggesting narrower hips relative to your height, leaning towards male.",
        'female': "Your hip breadth to stature ratio is high, suggesting wider hips relative to your height, leaning towards female."
    },
    'CS': {
        'male': "Your chest circumference is large relative to your height, leaning towards male.",
        'female': "Your chest circumference is small relative to your height, leaning towards female."
    },
    'FSR': {
        'male': "Your forearm circumference (flexed) is large relative to your height, leaning towards male.",
        'female': "Your forearm circumference (flexed) is small relative to your height, leaning towards female."
    },
    'CBR': {
        'male': "Your calf circumference is large relative to your buttock circumference, leaning towards male.",
        'female': "Your calf circumference is small relative to your buttock circumference, leaning towards female."
    },
    'BBSR': {
        'male': "Your shoulder breadth is wide relative to your height, leaning towards male.",
        'female': "Your shoulder breadth is narrow relative to your height, leaning towards female."
    },
    'BBHB': {
        'male': "Your shoulder breadth is wide relative to your hip breadth, leaning towards male.",
        'female': "Your shoulder breadth is narrow relative to your hip breadth, leaning towards female."
    },
    'ANKLS': {
        'male': "Your ankle circumference is large relative to your height, leaning towards male.",
        'female': "Your ankle circumference is small relative to your height, leaning towards female."
    },
    'FLS': {
        'male': "Your forearm-hand length is long relative to your height, leaning towards male.",
        'female': "Your forearm-hand length is short relative to your height, leaning towards female."
    },
    'WCS': {
        'male': "Your wrist circumference is large relative to your height, leaning towards male.",
        'female': "Your wrist circumference is small relative to your height, leaning towards female."
    }
}

if st.button("Worm me!"):
    if any(value == 0.0 for value in measurements.values()):
        st.error("Please provide all measurements.")
    else:
        with st.spinner('Analyzing...'):
            #Calculate ratio.
            def calculate_user_ratio_features(measurements):
                measurements = {key: value * 10 for key, value in measurements.items()}
                for key in measurements:
                    if measurements[key] == 0:
                        measurements[key] = 0.1  

                ratio_features = {
                    'WHR': measurements["waistcircumference"] / measurements["buttockcircumference"],
                    'HBS': measurements["hipbreadth"] / measurements["stature"],
                    'CS': measurements["chestcircumference"] / measurements["stature"],
                    'FSR': measurements["forearmcircumferenceflexed"] / measurements["stature"],
                    'CBR': measurements["calfcircumference"] / measurements["buttockcircumference"],
                    'BBSR': measurements["biacromialbreadth"] / measurements["stature"],
                    'BBHB': measurements["biacromialbreadth"] / measurements["hipbreadth"],
                    'ANKLS': measurements["anklecircumference"] / measurements["stature"],
                    'FLS': measurements["forearmhandlength"] / measurements["stature"],
                    'WCS': measurements["wristcircumference"] / measurements["stature"]
                }
                return ratio_features

            user_ratios = calculate_user_ratio_features(measurements)
            user_input = {**user_ratios, 'stature': measurements["stature"] * 10}
            user_df = pd.DataFrame([user_input])

            #Probability.
            passing_probability = model.predict_proba(user_df)[0][0] * 100  # Assuming class 0 is female
            st.subheader("Prediction Result")

            #Result color based on probability.
            if passing_probability < 50:
                result_color = "#e74c3c"  # Red
                extra_text = "You're a certified HON!"
                extra_color = "#c0392b"  # Deeper red
            elif 50 <= passing_probability < 80:
                result_color = "#f1c40f"  # Yellow
                extra_text = "Meh, twinkhon"
                extra_color = "#d4ac0d"  # Deeper yellow
            else:
                result_color = "#2ecc71"  # Green
                extra_text = "Passoid"
                extra_color = "#27ae60"
            
            st.markdown(f"""
            <div style='text-align: center; font-size: 24px; font-weight: bold; color: {result_color};'>
                Probability of passing as female: {passing_probability:.2f}%
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='text-align: center; font-size: 20px; font-weight: bold; color: {extra_color};'>
                {extra_text}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style='text-align: center; font-size: 14px; font-style: italic; color: #555555; margin-top: 10px;'>
                The interpretation might be more important than the score.
            </div>
            """, unsafe_allow_html=True)
        #SHAP
        shap_values = explainer.shap_values(user_df)
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Feature Interpretations")
            for feature_name, shap_value in zip(user_df.columns, shap_values[0]):
        # Get the full description of the feature
                if feature_name in ratio_descriptions:
                    full_name = f"{feature_name} ({ratio_descriptions[feature_name]})"
                else:
                    full_name = feature_name

        # Determine the direction
                direction = "male" if shap_value > 0 else "female"

                if feature_name == 'stature':
                # Convert 'stature' from mm to cm
                    feature_value_cm = user_df[feature_name].iloc[0] / 10  # Convert mm to cm
            # Basic interpretation
                    interpretation = f"""
                    <p style='color: var(--text-color);'>
                    <strong>{full_name}</strong>: {feature_value_cm:.2f} cm, leaning towards <strong>{direction}</strong> by {abs(shap_value):.2f}
                    </p>
                    """
                    st.markdown(interpretation, unsafe_allow_html=True)

            # Add 'Heighthon' interpretation if stature > 178 cm
                    if feature_value_cm > 178:
                        st.markdown("<p style='color: var(--text-color);'>Heighthon</p>", unsafe_allow_html=True)
                else:
                # For other features
                    feature_value = user_df[feature_name].iloc[0]
                # Basic interpretation
                    interpretation = f"""
                    <p style='color: var(--text-color);'>
                    <strong>{full_name}</strong>: {feature_value:.2f}, leaning towards <strong>{direction}</strong> by {abs(shap_value):.2f}
                    </p>
                    """
                    st.markdown(interpretation, unsafe_allow_html=True)

            # Add detailed interpretation if available
                    if feature_name in interpretation_templates:
                        detailed_interpretation = interpretation_templates[feature_name][direction]
                        detailed_text = f"<p style='color: var(--text-color);'>{detailed_interpretation}</p>"
                        st.markdown(detailed_text, unsafe_allow_html=True)

        with col_right:
            st.subheader("Feature Impact on Prediction")
            plt.figure(figsize=(8, 6))
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=user_df.iloc[0],
                    feature_names=user_df.columns
                ),
                max_display=len(user_df.columns),
                show=False
            )
            st.pyplot(plt)
            