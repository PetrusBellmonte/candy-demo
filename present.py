
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import norm

df = pd.read_csv('candy-data.csv')
st.set_page_config(page_title="Candy Data Analysis", layout="wide", page_icon="🍬")
st.title('Candy Dataset Overview')

st.header('First 10 Rows of the Dataset')
st.dataframe(df.head(10))

# Features to analyze (binary columns)
features = [
    'chocolate', 'fruity', 'caramel', 'peanutyalmondy',
    'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus'
]

# Absolute and relative occurrence
occurrence = pd.DataFrame({
    'Absolute': df[features].sum(),
    'Relative (%)': df[features].mean() * 100
})
st.header('Feature Occurrence (Absolute & Relative)')
st.dataframe(occurrence)
# General motivation for interpretation
'''_Given the profit focused nature of the (candy) market commonly appearing features, should be considered for the sole reason of having been chosen by more grounded market research.
Sure, some products result from historic development (and taste) however natural/economic selection should still result in strong presence "good" features._'''
# Interpretation
'''
_This would suggest, that chocolate and fruity flavors are particularly well-suited for the current market._
'''
# Caveats
'''
_It would also suggests that candy-mixes (=pluribus) are very popular. Ignoring that the candy selection is unknown, possibly skewing the occurrence distribution, pluribus occurrence can be inflated: A pluribus-unit of candy would spawn all its component candies, potentially leading to higher observed in the dataset._
'''
# Co-occurrence matrix
co_occurrence = df[features].T.dot(df[features])
st.header('Feature Co-occurrence Matrix')
st.dataframe(co_occurrence)

# Visualize co-occurrence as a heatmap
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
    z=co_occurrence.values,
    x=co_occurrence.columns,
    y=co_occurrence.index,
    colorscale='Blues',
    text=co_occurrence.values,
    hovertemplate='Feature 1: %{y}<br>Feature 2: %{x}<br>Count: %{z}<extra></extra>'
))
fig.update_layout(height=600, width=800)
st.plotly_chart(fig)


# Find strongest co-occurrences (excluding diagonal)
strongest_pairs = co_occurrence.where(~np.eye(len(features), dtype=bool))
strongest_pairs = strongest_pairs.div(strongest_pairs.sum(axis=1), axis=0)
strongest_pairs = strongest_pairs.stack().sort_values(ascending=False)
strongest_pairs = strongest_pairs.reset_index()
strongest_pairs.columns = ['Feature 1', 'Feature 2', 'Relative Strength']
# Compute Reverse Strength using a join with itself
strongest_pairs = strongest_pairs.merge(
    strongest_pairs,
    left_on=['Feature 1', 'Feature 2'],
    right_on=['Feature 2', 'Feature 1'],
    suffixes=('', '_reverse'),
    indicator=False,
    ).drop(['Feature 1_reverse', 'Feature 2_reverse'], axis=1)
strongest_pairs = strongest_pairs.rename(columns={'Relative Strength_reverse': 'Reverse Strength'})
strongest_pairs['Absolute Count'] = strongest_pairs.apply(
    lambda row: co_occurrence.loc[row['Feature 1'], row['Feature 2']], axis=1
)
top_n = st.slider("Show top N strongest feature co-occurrences", min_value=1, max_value=len(strongest_pairs), value=5)
st.subheader(f"Top {top_n} Strongest Feature Co-occurrences")
st.write("Relative Strength indicates how often Feature 2 appears when Feature 1 is present, relative to all occurrences of Feature 1. Reverse Strength shows the same relationship in the opposite direction.")
st.dataframe(
    strongest_pairs.head(top_n),hide_index=True
)
st.write("_Fruity and Pluribus co-occur frequently with strong relative strength in both directions._")

'# Win distributions'
'The dataset only provides winpercent as a measure of popularity, without raw vote counts or distribution details. This limits the depth of statistical analysis possible, as we cannot assess variance or confidence intervals for winpercent values.'
st.subheader("Top 10 Candies by Win Percent")
top5 = df.sort_values('winpercent', ascending=False).head(10)
st.dataframe(top5[['competitorname', 'winpercent']+features], hide_index=True)

st.subheader('Win Percent Distribution by Feature')
selected_feature = st.selectbox("Select a feature to plot winpercent histogram", features)
fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
for val in [0, 1]:
    subset = df[df[selected_feature] == val]['winpercent']
    ax_hist.hist(subset, bins=15, alpha=0.6, label=f"{selected_feature}={val}")
ax_hist.set_xlabel('Win Percent')
ax_hist.set_ylabel('Count')
ax_hist.set_title(f'Win Percent Distribution by {selected_feature}')
ax_hist.legend()
st.pyplot(fig_hist)

'''
_A look at the distribution of winratios for candy having a feature compared to not having it hints at:_
- _Chocolate is clearly better to have than not have._
- _Data on nougat and crispywafers is limited, but suggests potential benefits._
- _A bar is not a guarantee of quality, but seem to be an attractive formfactor unlike hard candy._
- _Neither mixes or fruits are popular. The similar results are expected due to their strong co-occurrence._
- _Caramel displays a rather huge variance in win ratios, indicating that its presence can significantly impact a candy's success._

'''
with st.expander("Gaussian Fit to Win Percent Distributions by Feature"):
    fig_gauss, ax_gauss = plt.subplots(figsize=(8, 6))

    x_vals = np.linspace(df['winpercent'].min(), df['winpercent'].max(), 200)
    for feature in features:
        subset = df[df[feature] == val]['winpercent']
        if len(subset) > 1:
            mu, std = subset.mean(), subset.std()
            pdf = norm.pdf(x_vals, mu, std)
            label = f"{feature} (μ={mu:.1f}, σ={std:.1f})"
            ax_gauss.plot(x_vals, pdf, label=label)
    ax_gauss.set_xlabel('Win Percent')
    ax_gauss.set_ylabel('Density')
    ax_gauss.set_title('Gaussian Fits to Win Percent by Feature')
    ax_gauss.legend(fontsize=8)
    st.pyplot(fig_gauss)

st.header("Variance of Win Percent for Identical Feature Profiles")

profile_cols = features
profile_groups = df.groupby(profile_cols)['winpercent']

identical_profiles = profile_groups.agg(['count', 'mean', 'std']).reset_index()
identical_profiles = identical_profiles[identical_profiles['count'] > 1]

st.write(f"Number of unique profiles with more than one candy: {len(identical_profiles)}")
st.dataframe(
    identical_profiles.sort_values('std', ascending=False),
    hide_index=True
)
st.write("_Variance (std) in winpercent for candies with identical profiles highlights possible effects of branding, marketing, implementation or other unmeasured factors._")


'# Regression'

st.header('Feature & Combination Counts')
with st.expander("Show thresholds", expanded=False):
    cols = st.columns(2)
    min_label_count = cols[0].number_input(
        'Minimum number of candies for a single feature (diagonal threshold)',
        min_value=1, max_value=int(co_occurrence.values.max()), value=3, step=1
    )
    min_pair_count = cols[1].number_input(
        'Minimum number of candies for a feature combination (off-diagonal threshold)',
        min_value=1, max_value=int(co_occurrence.values.max()), value=10, step=1
    )

# Single features and pairs meeting thresholds
used_features = {(feature,):co_occurrence.loc[feature, feature] for feature in features if co_occurrence.loc[feature, feature] >= min_label_count}
feature_pairs = {tuple(sorted((k1,k2))):v for (k1,k2),v in co_occurrence.where(co_occurrence >= min_pair_count).stack().items() if k1 != k2}
# I needed that :D
# co_occurrence.where(co_occurrence > min_pair_count).stack().items()

st.write(f"Used features (meeting diagonal threshold ≥ {min_label_count}): {len(used_features)} (of {len(features)})")
with st.expander("Show used features", expanded=False):
    st.dataframe(pd.DataFrame(list(used_features.items()), columns=['Features', 'Count']), hide_index=True)


st.write(f"Feature pairs (meeting off-diagonal threshold ≥ {min_pair_count}): {len(feature_pairs)}")
with st.expander("Show feature pairs", expanded=False):
    st.dataframe(pd.DataFrame(list(feature_pairs.items()), columns=['Features', 'Count']), hide_index=True)

# Triple combinations
three_feature_combos = {}
enable_three_combos = st.checkbox("Enable 3-feature combinations", value=True)
if enable_three_combos:


    for combo in combinations(features, 3):
        count = df[list(combo)].all(axis=1).sum()
        if count >= min_pair_count:
            three_feature_combos[tuple(sorted(combo))] = int(count)

    st.write(f"3-feature combinations (all present, count ≥ {min_pair_count}): {len(three_feature_combos)}")
    with st.expander("Show 3-feature combinations", expanded=False):
        st.dataframe(pd.DataFrame(list(three_feature_combos.items()), columns=['Features', 'Count']), hide_index=True)

st.header("Regression: Predicting Win Percent from Features")

X = df[features + ['sugarpercent', 'pricepercent']]
y = df['winpercent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_train_pred = reg.predict(X_train)

st.write(f"Test R² score: {r2_score(y_test, y_pred):.3f}")
st.write(f"Train R² score: {r2_score(y_train, y_train_pred):.3f}")

coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': reg.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

st.subheader("Feature Coefficients (Linear Regression)")
st.dataframe(coeffs, hide_index=True)

st.header("Regression: Including Selected Pairs and Triples")
# Add selected pairs and triples as new binary features
def make_combo_feature(df, combo):
    return df[list(combo)].all(axis=1).astype(int)

# Prepare new features
combo_features = []

# Add selected pairs
for pair in feature_pairs.keys():
    fname = f"{pair[0]}_{pair[1]}"
    df[fname] = make_combo_feature(df, pair)
    combo_features.append(fname)

# Add selected triples if enabled
if enable_three_combos:
    for triple in three_feature_combos.keys():
        fname = f"{triple[0]}_{triple[1]}_{triple[2]}"
        df[fname] = make_combo_feature(df, triple)
        combo_features.append(fname)

# Regression with extended features
all_features = features + ['sugarpercent', 'pricepercent'] + combo_features
X_ext = df[all_features]
y_ext = df['winpercent']

with st.expander("Model Selection & Parameters", expanded=True):
    model_type = st.radio("Choose regression model", ["Linear Regression", "Lasso"], index=1)
    test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
    if model_type == "Lasso":
        alpha_val = st.number_input("Lasso alpha", min_value=0.01, max_value=10.0, value=1.15, step=0.01)
        reg_ext = Lasso(alpha=alpha_val)
    else:
        reg_ext = LinearRegression()
use_only_top_features = st.checkbox("Use only top features from original regression", value=False)
if use_only_top_features:
    # SelectKBest to select top 10 features for regression with extended features
    from sklearn.feature_selection import SelectKBest, f_regression
    max_k = min(15, X_ext.shape[1])
    k_val = st.slider("Select number of top features (k) for regression", min_value=1, max_value=max_k, value=max_k)
    selector_ext = SelectKBest(score_func=f_regression, k=k_val)
    X_ext_new = selector_ext.fit_transform(X_ext, y_ext)
    selected_ext_features = [f for f, sel in zip(X_ext.columns, selector_ext.get_support()) if sel]
    st.write('Selected Features for Extended Regression:', selected_ext_features)
    X_ext_selected = X_ext[selected_ext_features]
else:
    X_ext_selected = X_ext

X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_ext_selected, y_ext, test_size=test_size, random_state=42)
reg_ext.fit(X_train_ext, y_train_ext)
y_pred_ext = reg_ext.predict(X_test_ext)
y_train_pred_ext = reg_ext.predict(X_train_ext)

st.write(f"Test R² score (extended{", top 10 features" if use_only_top_features else ""}): {r2_score(y_test_ext, y_pred_ext):.3f}")
st.write(f"Train R² score (extended{", top 10 features" if use_only_top_features else ""}): {r2_score(y_train_ext, y_train_pred_ext):.3f}")

coeffs_ext = pd.DataFrame({
    'Feature': X_ext_selected.columns,
    'Coefficient': reg_ext.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

st.subheader(f"Feature Coefficients (Extended Regression{", Top 10 Features" if use_only_top_features else ""})")
st.dataframe(coeffs_ext, hide_index=True)

'''
#### The fruit mix situation:
_Modeling with linear regression, the pluribus is heavily punished, expect for hard fruity mixes which are... fine? This hints at candy, that is served as a part of mixes generally being perceived as less favorable._
#### Chocolate
_Chocolate consistently performs well, improved by a bar form with peanut-almondy something._
'''

st.subheader("Chocolaty, Peanut-Almondy Bars")
choco_peanut_bar = df[
    (df['chocolate'] == 1) &
    (df['peanutyalmondy'] == 1) &
    (df['bar'] == 1)
][['competitorname','winpercent'] + [f for f in features if f not in ['competitorname','winpercent','chocolate','peanutyalmondy','bar']]]
st.dataframe(choco_peanut_bar, hide_index=True)

'# Summary'
'''
_Chocolate with some other taste-elements is the most reliable winner. 
The formfactor of a bar also reliably improves popularity. In regard to the additional favour(s) peanut-almondy takes the lead, however caramel, nougat and crispywafers have all been successfully implemented in the market.
Being part of a snackbox does not enhance desirability._
'''