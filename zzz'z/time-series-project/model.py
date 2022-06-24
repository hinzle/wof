from imports import *

# create a color palette for seaborn viz's that match our canva presentation template
canva_palette = ['#FFB000', '#375F00', '#707E00', '#B39B00', '#002729']

def create_projections_1(test):
    '''
    Creates a dataframe of actual and projected monthly profits for 2017. 
    Projected profits are based on a hypothetical duplication of sales in the 
    Technology category. 
    '''
    # create a df of additional sales by pulling all Technology sales from the test df
    hypo_tech_sales_17 = test[test.category == 'Technology']
    # concat those additional sales to the test dataframe
    hypothetical_17 = pd.concat([test, hypo_tech_sales_17])
    # createa an empty dataframe
    projections = pd.DataFrame()
    # add actual monthly profits to the projections df
    projections['actual_2017'] = test.resample('M')['profit'].sum()
    # add projected monthly profits to the projections df
    projections['projected_2017'] = hypothetical_17.resample('M')['profit'].sum()
    # add a column that is the difference between projected and actual profits
    projections['difference'] = projections.projected_2017 - projections.actual_2017
    # add a column that is the proportion of increase or decrease from actual to projected
    projections['difference_pct'] = projections.difference / projections.actual_2017
    # add cumulative sum columns for actual and projected
    projections['rolling_total_actual'] = projections.actual_2017.cumsum()
    projections['rolling_total_projected'] = projections.projected_2017.cumsum()
    # reformat index for easier visualizations
    projections.index = projections.index.strftime('%b')
    projections.index.name = 'month'
    
    return projections

def monthly_profit_barplot(projections):
    '''
    Creates a barplot of actual vs projected monthly profit for 2017 based on the projections df. 
    '''
    # melt the projections df for easier visualization
    projections_melted = projections[['actual_2017', 'projected_2017']].melt(value_name='sales', ignore_index=False)
    plt.figure(figsize=(12,8)) 
    # create the barplot
    sns.barplot(data=projections_melted, x=projections_melted.index, y='sales', hue='variable', 
                palette=canva_palette, ec='black')
    plt.title('Monthly Profit 2017', fontsize=24, color='#002729')
    plt.ylabel('Profit', fontsize=18, color='#002729')
    plt.xlabel(None)
    plt.xticks(fontsize=14, color='#002729')
    plt.yticks(fontsize=14, color='#002729')
    plt.legend(loc='lower left', fontsize=18)
    # get the current axis object (needed for the next line)
    ax = plt.gca()
    # reformat y tick labels as currency
    ax.set_yticklabels([f'${tick:,.0f}' for tick in ax.get_yticks()])
    # save the figure as jpeg to the working directory
    plt.savefig('fig_monthly_profit_barplot.jpeg')
    plt.show()

def create_projections_2(test):
    '''
    Creates a dataframe of actual and projected monthly profits for 2017. 
    Projected profits are based on a hypothetical duplication of sales in the 
    Technology category. 

    Repeats the process outlined in create_projections_1, but this time removes
    all sales with discounts greater than 35%.
    '''
    # create additional sales for 2017 by duplicating the existing sales
    hypo_tech_sales_17 = test[test.category == 'Technology']
    # remove sales with discounts > 35%
    hypo_tech_sales_17 = hypo_tech_sales_17[hypo_tech_sales_17.discount <= .35]
    # add the hypothetical tech sales to the existing sales from 2017
    hypothetical_17 = pd.concat([test, hypo_tech_sales_17])
    # remaining code repeats the process from create_projections_1
    projections = pd.DataFrame()
    projections['actual_2017'] = test.resample('M')['profit'].sum()
    projections['projected_2017'] = hypothetical_17.resample('M')['profit'].sum()
    projections['difference'] = projections.projected_2017 - projections.actual_2017
    projections['difference_pct'] = projections.difference / projections.actual_2017
    projections['rolling_total_actual'] = projections.actual_2017.cumsum()
    projections['rolling_total_projected'] = projections.projected_2017.cumsum()
    projections.index = projections.index.strftime('%b')
    projections.index.name = 'month'
    
    return projections

def total_profit_lineplot(projections):
    '''
    Creates seaborn lineplot of cumulative sum of monthly profits in 2017, actual and projected. 
    Based on projections dataframe.
    '''
    plt.figure(figsize=(12,8))
    sns.lineplot(data=profits, x=profits.index, y='rolling_total_actual', 
                color='#FFB000', label='Actual', lw=6)
    sns.lineplot(data=profits, x=profits.index, y='rolling_total_projected', 
                color='#707E00', label='Projected', lw=6)
    plt.title('Total Profit 2017', fontsize=24, color='#002729')
    plt.ylabel('Total Profit', fontsize=18, color='#002729')
    plt.xlabel(None)
    plt.xticks(fontsize=14, color='#002729')
    plt.yticks(fontsize=14, color='#002729')
    plt.legend(loc='upper left', fontsize=18)
    ax = plt.gca()
    ax.set_yticklabels([f'${tick:,.0f}' for tick in ax.get_yticks()])
    plt.savefig('fig_total_profit_2017.jpeg')
    plt.show()