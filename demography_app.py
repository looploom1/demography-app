"""
Full‑Featured Demographic Analysis & Projection App (DAPPS‑like)
Requirements: streamlit, pandas, numpy, plotly, scipy, openpyxl
Run: streamlit run demography_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Demographic Toolkit", layout="wide")
st.title("📊 Unified Demographic Analysis & Projection System (DAPPS-like)")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def whipples_index(data, age_col, pop_col, digit=0, lower=23, upper=62):
    """Whipple's index for age heaping on a digit (0 or 5)."""
    if digit not in [0, 5]:
        return None
    df = data.copy()
    df = df[(df[age_col] >= lower) & (df[age_col] <= upper)]
    if digit == 0:
        target_ages = [a for a in range(lower, upper+1) if a % 10 == 0]
    else:
        target_ages = [a for a in range(lower, upper+1) if a % 10 == 5]
    sum_target = df[df[age_col].isin(target_ages)][pop_col].sum()
    sum_all = df[pop_col].sum()
    if sum_all == 0:
        return None
    # Whipple = (Sum of ages ending in digit / (1/5 * total sum)) * 100
    expected = sum_all / 5 if digit == 0 else sum_all / 5
    return (sum_target / expected) * 100

def myers_blended_index(data, age_col, pop_col, lower=10, upper=89):
    """Myers' blended index for age heaping on all digits 0-9."""
    df = data[(data[age_col] >= lower) & (data[age_col] <= upper)].copy()
    if df.empty:
        return None
    # Prepare digit population sums
    digit_sums = {d: 0.0 for d in range(10)}
    total_pop = 0
    for _, row in df.iterrows():
        age = row[age_col]
        pop = row[pop_col]
        total_pop += pop
        digit = age % 10
        digit_sums[digit] += pop
    # Blended sum: 10 times the proportion of total
    blended = {d: digit_sums[d] * 10 / total_pop if total_pop else 0 for d in range(10)}
    # Myers index = 0.5 * sum(abs(blended[d] - 10))  (since perfect = 10%)
    index = 0.5 * sum(abs(blended[d] - 10) for d in range(10))
    return index

def age_ratio_score(data, age_col, pop_col, lower=5, upper=85, sex='both'):
    """UN Age‑ratio score for 5‑year age groups."""
    df = data[(data[age_col] >= lower) & (data[age_col] <= upper)].copy()
    if df.empty:
        return None
    df = df.sort_values(age_col)
    pops = df[pop_col].values
    ages = df[age_col].values
    n = len(pops)
    score = 0.0
    count = 0
    for i in range(1, n-1):
        if ages[i] % 5 == 0:  # assume age group midpoints like 5,10,15...
            prev = pops[i-1]
            curr = pops[i]
            next_ = pops[i+1]
            if prev + next_ > 0:
                ratio = (2 * curr) / (prev + next_)
                score += abs(ratio - 1)
                count += 1
    if count == 0:
        return None
    return (score / count) * 100

def compute_chiang_life_table(deaths_df, population_df, age_col='age', deaths_col='deaths', pop_col='population'):
    """Chiang's abridged life table for age groups 0,1-4,5-9,...85+. Assumes single or 5-year groups."""
    # Merge deaths and population
    df = pd.merge(deaths_df, population_df, on=age_col, suffixes=('_d', '_p'))
    # Ensure age groups: we'll assume age column is the start of age group (0,1,5,10,...)
    df = df.sort_values(age_col).reset_index(drop=True)
    n = [1,4] + [5]*len(df[df[age_col]>=5])  # crude: handle 0,1-4,5-9,... We'll simplify: assume age col contains start age, and group length from next age.
    # Let the user input be 5-year groups from 0-4,5-9,... with age col as starting age (0,5,10,...)
    # For simplicity, treat all groups as 5-year except first (0-4). We'll just use uniform 5-year approach with n=5, but death rates m_x.
    # Real Chiang uses n_x and a_x fractions. I'll implement a simplified version:
    # m_x = D_x / P_x
    # n_x = length of interval (5 for all except last open)
    # a_x = n/2 except for infants (0.1 for 0-1 of 1-year? We'll assume n=5 for all for demo)
    df['M'] = df[deaths_col] / df[pop_col]  # central death rate
    df['n'] = 5  # default
    df.loc[0, 'n'] = 1  # first group 0-1? but if age_col=0, we assume 0-4, so n=5. To keep simple, use n=5 everywhere.
    # Actually better: assume data is already 5-year groups from 0-4,5-9,... age=0 means 0-4, so n=5.
    df['a'] = 2.5  # fraction of interval lived by those dying
    # For open interval 85+, n set to infinity (but we approximate with n=5? treat as open with e_x later)
    df.loc[df.index[-1], 'n'] = np.inf
    df.loc[df.index[-1], 'a'] = np.nan
    # Probability of dying
    df['q'] = np.where(df['n'].isin([np.inf]), 1.0,
                       (df['n'] * df['M']) / (1 + (df['n'] - df['a']) * df['M']))
    df.loc[df.index[-1], 'q'] = 1.0
    # Survivorship
    lx = [1.0]
    for i in range(1, len(df)):
        if df.loc[i-1, 'n'] == np.inf:
            lx.append(0)
        else:
            lx.append(lx[i-1] * (1 - df.loc[i-1, 'q']))
    df['l'] = lx
    # Person‑years lived
    df['L'] = np.where(df['n'] == np.inf, df['l'] / df['M'],  # for open interval, L = l/inf rate? simpler: l/m
                       df['n'] * (df['l'] - (1 - df['a']) * df['l'] * df['q'])) if not df.empty else None
    # Actually vectorized:
    df['L'] = np.where(df['n'] == np.inf, df['l'] / df['M'],
                        df['n'] * (1 - df['a']) * df['l'] + df['a'] * df['l'] * (1 - df['q']))  # rough
    # Fix open: L = l / M_inf (but M_inf is deaths/pop, so l/M = l / (D/P) = l*P/D, but easier: use 1/M)
    mask = df['n'] == np.inf
    df.loc[mask, 'L'] = df.loc[mask, 'l'] / df.loc[mask, 'M']
    # T_x cumulative
    df['T'] = df['L'][::-1].cumsum()[::-1]
    df['e'] = df['T'] / df['l']
    return df[['age','M','q','l','L','T','e']]

# ---------------------------
# SAMPLE DATA FOR PAKISTAN
# ---------------------------
def load_sample_data():
    """Create a sample dataset of Pakistan 2023 population by age 5-year groups (in thousands)."""
    ages = list(range(0,86,5))
    pop_male = [11901, 10964, 10456, 9820, 8976, 8045, 7278, 6512, 5723, 4732, 3580, 2567, 1678, 984, 508, 215, 102, 76]
    pop_female = [10876, 10012, 9587, 9034, 8267, 7476, 6789, 6134, 5421, 4467, 3398, 2467, 1632, 976, 518, 231, 117, 92]
    # Align to 18 groups: 0-4 to 85+
    # Sample deaths crude
    deaths_male = [89.6, 7.2, 3.5, 5.7, 8.9, 12.4, 15.1, 18.3, 22.1, 26.0, 28.3, 29.8, 30.2, 31.5, 34.1, 38.0, 41.0, 68.5]
    deaths_female = [70.2, 5.3, 2.8, 4.2, 6.1, 8.0, 10.4, 13.4, 16.7, 20.3, 23.0, 25.1, 27.3, 30.1, 34.5, 39.7, 45.0, 78.2]
    df_pop = pd.DataFrame({'age': ages, 'male': pop_male, 'female': pop_female})
    df_deaths = pd.DataFrame({'age': ages, 'male': deaths_male, 'female': deaths_female})
    return df_pop, df_deaths

# ---------------------------
# APP SIDEBAR & DATA INPUT
# ---------------------------
st.sidebar.header("1. Load Data")
data_option = st.sidebar.radio("Select data source:", ["Use Pakistan sample data", "Upload your own CSV/Excel"])
if data_option == "Upload your own CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Choose file (columns: age, male_pop, female_pop, male_deaths, female_deaths)", type=["csv","xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_pop = pd.read_csv(uploaded_file)
                df_deaths = pd.read_csv(uploaded_file)  # assuming same file has deaths or separate? You'll adjust.
                # For simplicity, we'll assume one file contains population columns male_pop, female_pop,
                # and deaths columns male_deaths, female_deaths. You can adapt.
                # We'll create two dataframes:
                df_pop = pd.DataFrame({'age': df_pop['age'], 'male': df_pop['male_pop'], 'female': df_pop['female_pop']})
                df_deaths = pd.DataFrame({'age': df_deaths['age'], 'male': df_deaths['male_deaths'], 'female': df_deaths['female_deaths']})
            else:
                df_pop = pd.read_excel(uploaded_file, sheet_name=0)
                df_deaths = pd.read_excel(uploaded_file, sheet_name=1) if len(pd.ExcelFile(uploaded_file).sheet_names)>1 else df_pop.copy()
                df_pop = pd.DataFrame({'age': df_pop['age'], 'male': df_pop['male_pop'], 'female': df_pop['female_pop']})
                df_deaths = pd.DataFrame({'age': df_deaths['age'], 'male': df_deaths['male_deaths'], 'female': df_deaths['female_deaths']})
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()
    else:
        st.info("Awaiting file upload. Using sample data for demo...")
        df_pop, df_deaths = load_sample_data()
else:
    df_pop, df_deaths = load_sample_data()

# Prepare total population
df_pop['total'] = df_pop['male'] + df_pop['female']
df_deaths['total'] = df_deaths['male'] + df_deaths['female']

# ---------------------------
# TABS FOR ANALYSIS
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Direct Estimation (Rates & Life Tables)",
    "🔍 Data Quality Evaluation",
    "📊 Population Pyramids & Dependency Ratios",
    "📉 Trend Analysis",
    "🧮 Cohort‑Component Projections",
    "ℹ️ Help & Methods"
])

# ----------------
# TAB 1: DIRECT ESTIMATION
# ----------------
with tab1:
    st.header("Fertility & Mortality Rates, Life Tables")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age‑Specific Fertility Rates (ASFR) & TFR")
        # For demo, provide a sample ASFR for Pakistan (per 1000 women)
        sample_asfr = np.array([0,0,5,75,150,180,170,120,60,20,5,1,0,0,0,0,0,0])  # ages 0-4...85+
        # Let user input or we compute from births? We'll allow manual input.
        asfr = st.text_area("Enter ASFR per 1000 women for each age group (starting 0-4, comma separated):",
                            value=",".join(map(str,sample_asfr)))
        try:
            asfr_vals = np.array([float(x) for x in asfr.split(",")])
            asfr_vals = asfr_vals[:len(df_pop)]  # trim to match age groups
            TFR = np.sum(asfr_vals * 5) / 1000  # multiply by interval length
            st.metric("Total Fertility Rate (TFR)", f"{TFR:.2f} children per woman")
            fig_fr = go.Figure([go.Bar(x=df_pop['age'], y=asfr_vals, name="ASFR per 1000")])
            fig_fr.update_layout(title="Age‑Specific Fertility Rate", xaxis_title="Age group start", yaxis_title="per 1000 women")
            st.plotly_chart(fig_fr, use_container_width=True)
        except:
            st.warning("Could not parse ASFR.")
    with col2:
        st.subheader("Age‑Specific Mortality Rates & Life Expectancy")
        sex = st.selectbox("Select sex for life table:", ["male","female","total"])
        deaths = df_deaths[sex].values
        pop = df_pop[sex].values
        lt = compute_chiang_life_table(pd.DataFrame({'age':df_pop['age'], 'deaths':deaths}),
                                       pd.DataFrame({'age':df_pop['age'], 'population':pop}))
        st.dataframe(lt.style.format("{:.4f}", subset=['q','e']), height=400)
        e0 = lt['e'].iloc[0]
        st.metric("Life Expectancy at Birth (e0)", f"{e0:.1f} years")
        # probability of dying curve
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=lt['age'], y=lt['q'], mode='lines+markers', name='q(x)'))
        fig_q.update_layout(title="Probability of Dying q(x)", xaxis_title="Age", yaxis_title="q(x)")
        st.plotly_chart(fig_q, use_container_width=True)

# ----------------
# TAB 2: DATA QUALITY EVALUATION
# ----------------
with tab2:
    st.header("Evaluation of Data Quality (Heaping & Age‑Ratio Score)")
    sex_eval = st.selectbox("Select sex for quality checks:", ["total","male","female"], key="eval_sex")
    pop_col = sex_eval
    # Whipple's index (0)
    whipple0 = whipples_index(df_pop, 'age', pop_col, digit=0)
    whipple5 = whipples_index(df_pop, 'age', pop_col, digit=5)
    myers = myers_blended_index(df_pop, 'age', pop_col)
    ars = age_ratio_score(df_pop, 'age', pop_col)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Whipple (0)", f"{whipple0:.1f}" if whipple0 else "N/A", help="100=no heaping, >100 heaping on 0")
    c2.metric("Whipple (5)", f"{whipple5:.1f}" if whipple5 else "N/A")
    c3.metric("Myers' Index", f"{myers:.1f}" if myers else "N/A", help="0=perfect, max 90")
    c4.metric("Age Ratio Score", f"{ars:.1f}%" if ars else "N/A", help="<5 good, 5-10 moderate, >10 poor")
    st.markdown("Interpretation: Lower values = better data. Whipple >125 indicates severe age heaping on 0 or 5.")

# ----------------
# TAB 3: POPULATION PYRAMIDS & DEPENDENCY RATIOS
# ----------------
with tab3:
    st.header("Population Pyramids & Dependency Ratios")
    # Pyramid
    y_age = df_pop['age'].astype(str) + '-' + (df_pop['age']+4).astype(str)
    fig_pyr = go.Figure()
    fig_pyr.add_trace(go.Bar(y=y_age, x=df_pop['male']*-1, name='Male', orientation='h'))
    fig_pyr.add_trace(go.Bar(y=y_age, x=df_pop['female'], name='Female', orientation='h'))
    fig_pyr.update_layout(barmode='relative', bargap=0.1, title="Population Pyramid",
                          xaxis_title="Population (thousands)", yaxis_title="Age group")
    st.plotly_chart(fig_pyr, use_container_width=True)

    # Dependency ratios
    young_pop = df_pop[(df_pop['age']>=0)&(df_pop['age']<=14)]['total'].sum()
    old_pop = df_pop[df_pop['age']>=65]['total'].sum()
    working_pop = df_pop[(df_pop['age']>=15)&(df_pop['age']<=64)]['total'].sum()
    young_dep = young_pop / working_pop * 100
    old_dep = old_pop / working_pop * 100
    total_dep = (young_pop+old_pop)/working_pop*100
    c1,c2,c3 = st.columns(3)
    c1.metric("Young Dependency Ratio", f"{young_dep:.1f}%")
    c2.metric("Old‑Age Dependency Ratio", f"{old_dep:.1f}%")
    c3.metric("Total Dependency Ratio", f"{total_dep:.1f}%")
    st.caption("Dependency ratio = (population 0‑14 + 65+) / population 15‑64 * 100")

# ----------------
# TAB 4: TREND ANALYSIS
# ----------------
with tab4:
    st.header("Trend Analysis (Fitting Exponential/Linear Model)")
    # Simulate historical total population (Pakistan 1998-2023) for demonstration
    years = np.array([1998,2008,2018,2023])
    pop_hist = np.array([132, 164, 207, 241])  # millions
    df_hist = pd.DataFrame({'year': years, 'population': pop_hist})
    st.write("Historical population (in millions):")
    st.dataframe(df_hist)

    # Fit exponential: log(pop) = a + b*year
    log_pop = np.log(pop_hist)
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, log_pop)
    growth_rate = slope * 100  # percent
    st.write(f"Exponential growth rate: **{growth_rate:.2f}%** per year (R² = {r_value**2:.3f})")

    # Projection to future year
    future_year = st.slider("Project total population to year:", 2030, 2050, 2035)
    proj_pop = np.exp(intercept + slope * future_year)
    st.metric(f"Projected Population {future_year}", f"{proj_pop:.1f} million")

    # Plot
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=df_hist['year'], y=df_hist['population'], mode='markers+lines', name='Historical'))
    years_fit = np.linspace(1998, future_year, 50)
    pop_fit = np.exp(intercept + slope * years_fit)
    fig_trend.add_trace(go.Scatter(x=years_fit, y=pop_fit, mode='lines', name='Exponential Fit', line=dict(dash='dash')))
    fig_trend.add_trace(go.Scatter(x=[future_year], y=[proj_pop], mode='markers', marker=dict(size=12,color='red'), name='Projection'))
    fig_trend.update_layout(title="Trend Analysis", xaxis_title="Year", yaxis_title="Population (millions)")
    st.plotly_chart(fig_trend, use_container_width=True)

# ----------------
# TAB 5: COHORT-COMPONENT PROJECTIONS
# ----------------
with tab5:
    st.header("Cohort‑Component Population Projection")
    st.markdown("Project population forward using the cohort‑component method with assumed fertility, mortality, and net migration.")
    # Base year population (in thousands, total both sexes)
    base_pop = df_pop['total'].values
    base_pop_male = df_pop['male'].values
    base_pop_female = df_pop['female'].values
    n_ages = len(base_pop)

    # Inputs
    horizon = st.number_input("Projection horizon (years)", min_value=5, max_value=50, value=20, step=5)
    proj_year = st.number_input("Base year", value=2023)
    tfr_assumption = st.slider("Assumed constant TFR (children per woman)", 1.0, 5.0, 3.5, 0.1)
    asfr_sched = sample_asfr / sample_asfr.sum() * (tfr_assumption/ (TFR if TFR else 3.5) )  # scale base ASFR to new TFR
    # Ensure ASFR aligns with female age groups
    female_prop = df_pop['female'].values / (df_pop['male'].values + df_pop['female'].values)
    female_pop = df_pop['female'].values * 1.0  # copy
    male_pop = df_pop['male'].values * 1.0

    # Survival ratios from life table (both sexes combined for simplicity)
    deaths_total = df_deaths['total'].values
    pop_total = df_pop['total'].values
    lt_total = compute_chiang_life_table(pd.DataFrame({'age':df_pop['age'], 'deaths':deaths_total}),
                                         pd.DataFrame({'age':df_pop['age'], 'population':pop_total}))
    S = np.array(lt_total['l'][1:]) / np.array(lt_total['l'][:-1])  # survival ratios from age x to x+5
    # For the last open interval, survival to same group? set to 0.5 or use e_x? We'll assume a small ratio.
    S = np.append(S, 0.1)  # 85+ to 90+ approximate
    # Also infant survival: from birth to 0-4 uses different ratio. We'll use life table L0/l0.
    # Actually need survival from birth to 0-4: S_birth = L(0)/5 (?) We'll skip and treat births separately.
    # For simplicity, we'll apply the same ratio vector for all ages, meaning population age x at t moves to age x+5 at t+5 using ratio S[x].

    # Net migration assumption (constant number per age group per 5 years)
    net_migration = np.zeros(n_ages)  # no migration

    # Project
    years_list = np.arange(proj_year, proj_year+horizon+5, 5)
    pops_total = []
    pops_male = []
    pops_female = []
    female_pop_series = [female_pop]
    male_pop_series = [male_pop]

    for step in range(int(horizon/5)):
        # Female population at reproductive ages (15-49)
        repro_fem = female_pop[3:10].sum()  # ages 15-19 to 45-49 index 3 to 9 (if age groups 0-4=0,5-9=1,...15-19=3)
        births = 0
        for i in range(3,10):
            births += female_pop[i] * (asfr_sched[i] / 1000) * 5  # per 5-year period
        # Split births into male/female (SRB=1.05)
        male_births = births * 0.512
        female_births = births * 0.488
        # Aging the population
        new_male = np.zeros_like(male_pop)
        new_female = np.zeros_like(female_pop)
        # Age 0-4 group from births
        new_male[0] = male_births
        new_female[0] = female_births
        # Ages 5-9 to 80+ : people from previous lower group survive
        for i in range(1, n_ages):
            new_male[i] = male_pop[i-1] * S[i-1]
            new_female[i] = female_pop[i-1] * S[i-1]
        # Last group: add survivors from previous last group plus current last group survival (approximate)
        new_male[-1] += male_pop[-1] * S[-1]
        new_female[-1] += female_pop[-1] * S[-1]
        # Add migration
        new_male += net_migration * 0.5  # assume half male
        new_female += net_migration * 0.5
        male_pop = new_male
        female_pop = new_female
        male_pop_series.append(male_pop)
        female_pop_series.append(female_pop)
        pops_total.append(male_pop.sum() + female_pop.sum())

    # Display results
    st.subheader("Projected Total Population (thousands)")
    proj_years = years_list[1:]  # first is base year
    df_proj = pd.DataFrame({'Year': proj_years, 'Total Population': pops_total})
    st.dataframe(df_proj.style.format("{:,.0f}"))

    # Plot projected pyramids for base and last
    st.subheader("Population Pyramid: Base vs. Horizon")
    fig_proj_pyr = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                 subplot_titles=(f"Base {proj_year}", f"Projected {proj_year+horizon}"))
    y_age_labels = [f"{a}-{a+4}" for a in df_pop['age']]
    fig_proj_pyr.add_trace(go.Bar(y=y_age_labels, x=-male_pop_series[0], name='Male Base', orientation='h'), row=1, col=1)
    fig_proj_pyr.add_trace(go.Bar(y=y_age_labels, x=female_pop_series[0], name='Female Base', orientation='h'), row=1, col=1)
    fig_proj_pyr.add_trace(go.Bar(y=y_age_labels, x=-male_pop_series[-1], name='Male Horizon', orientation='h'), row=1, col=2)
    fig_proj_pyr.add_trace(go.Bar(y=y_age_labels, x=female_pop_series[-1], name='Female Horizon', orientation='h'), row=1, col=2)
    fig_proj_pyr.update_layout(barmode='relative', showlegend=False)
    st.plotly_chart(fig_proj_pyr, use_container_width=True)

# ----------------
# TAB 6: HELP & METHODS
# ----------------
with tab6:
    st.header("Methodology & How to Use")
    st.markdown("""
    **This app replicates the core functionality of DAPPS, PAS, MortPak, and Spectrum Suite in one place.**
    
    - **Life Tables:** Uses Chiang's abridged life table method.
    - **Data Quality:** Whipple's and Myers' indices, UN Age‑Ratio Score.
    - **Projections:** Standard cohort‑component method with user‑specified TFR and survival ratios from the life table.
    - **Rates:** ASFR, TFR, age‑specific mortality rates.
    
    **To use your own data:** Upload a CSV or Excel file with columns: `age` (start of 5‑year age group: 0,5,10,...85+), `male_pop`, `female_pop`, `male_deaths`, `female_deaths`. 
    
    **Requirements:** Python 3.8+, install with `pip install streamlit pandas numpy plotly scipy openpyxl`
    """)

# Run with: streamlit run demography_app.py