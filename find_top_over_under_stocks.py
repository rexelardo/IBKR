import pandas as pd
df = pd.read_csv('tickers.csv')


def get_top_n_companies(group, n=3):
    return group.sort_values(by='marketCap', ascending=False).head(n)

# Calculate total market cap for each industry
industry_total_market_cap = df.groupby('industry')['marketCap'].sum().sort_values(ascending=False)

# Initialize an empty list to store results
sorted_top_companies_per_industry_list = []

# Iterate through industries sorted by their total market cap
for industry_name in industry_total_market_cap.index:
    # Filter the original DataFrame for the current industry
    industry_df = df[df['industry'] == industry_name]
    # Get the top companies for this industry
    top_companies = get_top_n_companies(industry_df)
    # Append to the list
    sorted_top_companies_per_industry_list.append(top_companies)

# Concatenate all results into a single DataFrame
sorted_top_companies_per_industry = pd.concat(sorted_top_companies_per_industry_list)

print('Top 3 companies by market cap per industry (industries sorted by total market cap) complete')
# display(sorted_top_companies_per_industry.head(20))
sorted_top_companies_per_industry.to_csv('top_companies_per_industry.csv', index=False)


def get_bottom_n_companies(group, n=3):
    return group.sort_values(by='marketCap', ascending=True).head(n)

# Initialize an empty list to store results for smallest companies
sorted_bottom_companies_per_industry_list = []

# Iterate through industries sorted by their total market cap (to maintain the same industry order)
for industry_name in industry_total_market_cap.index:
    # Filter the original DataFrame for the current industry
    industry_df = df[df['industry'] == industry_name]
    # Get the bottom companies for this industry
    bottom_companies = get_bottom_n_companies(industry_df)
    # Append to the list
    sorted_bottom_companies_per_industry_list.append(bottom_companies)

# Concatenate all results into a single DataFrame
sorted_bottom_companies_per_industry = pd.concat(sorted_bottom_companies_per_industry_list)

print('Top 3 smallest companies by market cap per industry (industries sorted by total market cap) complete')
# display(sorted_bottom_companies_per_industry.head(20))
sorted_bottom_companies_per_industry.to_csv('bottom_companies_per_industry.csv', index=False)