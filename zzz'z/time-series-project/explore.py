from imports import *

# create a custom seaborn color palette to match the canva presentation theme
canva_palette = ['#FFB000', '#375F00', '#707E00', '#B39B00', '#002729']

def sales_volume_by_category(train):
    '''
    Takes in the train sample df and 
    outputs a seaborn barplot of total quantity by category.
    '''
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='category', y='quantity', 
                estimator=sum, ci=None, ec='black', palette=canva_palette)
    plt.title('Sales Volume by Category', fontsize=24)
    plt.ylabel('Total Sales Volume', fontsize=18)
    plt.xlabel('Product Category', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('sales_volume_by_category.jpeg')
    plt.show()

def profit_per_product_by_category(train):
    '''
    Takes in the train sample df and outputs
    a seaborn barplot of average profit-per-product by category.
    '''
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='category', y='profit_per_product', 
                ci=None, ec='black', palette=canva_palette)
    plt.title('Average Profit-per-Product by Category', fontsize=24)
    plt.ylabel('Average Profit-per-Product', fontsize=18)
    plt.xlabel('Product Category', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('profit_per_product_by_category.jpeg')
    plt.show()