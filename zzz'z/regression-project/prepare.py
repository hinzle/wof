from imports import *

def ml_data(train, validate, test, target=list):
	'''
	->: train, validate, test 
	<-: X_train, y_train, X_validate, y_validate, X_test, y_test
	'''
	X_train = train.drop(columns=target)
	y_train = train[target]
	X_validate = validate.drop(columns=target)
	y_validate = validate[target]
	X_test = test.drop(columns=target)
	y_test = test[target]
	return [X_train, y_train, X_validate, y_validate, X_test, y_test]

def residuals(actual, predicted):
    '''
    ∆(y,yhat)
    '''
    return actual - predicted

def sse(actual, predicted):
    '''
    sum of squared error
    '''
    return (residuals(actual, predicted) ** 2).sum()

def mse(actual, predicted):
    '''
    mean squared error
    '''
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    '''
    root mean squared error
    '''
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    '''
    explained sum of squared error
    '''
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    '''
    total sum of squared error
    '''
    return ((actual - actual.mean()) ** 2).sum()

def r2_score(actual, predicted):
    '''
    explained variance
    '''
    return ess(actual, predicted) / tss(actual)

def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
        'r2': r2_score(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    sse_baseline = sse(actual, actual.mean())
    sse_model = sse(actual, predicted)
    return sse_model < sse_baseline

def evaluate_hypothesis(p: float, alpha: float = 0.05, output: bool = True) -> bool:
    '''
    Compare the p value to the established alpha value to determine if the null hypothesis
    should be rejected or not.
    '''

    if p < alpha:
        if output:
            print('\nReject H_0')
        return False
    else: 
        if output:
            print('\nFail to Reject H_0')
        return True

def pearsons_r_p(data_for_category1, data_for_category2, alpha = 0.05):
    '''
    Given two subgroups from a dataset, conducts a correlation test for linear relationship and outputs 
    the relevant information to the console. 
    Utilizes the method provided in the Codeup curriculum for conducting correlation test using
    scipy and pandas.

    "  
    '''

    # conduct test using scipy.stats.peasonr() test
    r, p = stats.pearsonr(data_for_category1, data_for_category2)

    # output
    print(f'r = {r:.4f}')
    print(f'p = {p:.4f}')

    # evaluate the hypothesis against the established alpha value
    evaluate_hypothesis(p, alpha)

    if r<=0.2:
        print("no correlation")
    elif r<=0.5:
        print("weak correlation")
    elif r<=0.75:
        print("moderate correlation")
    else:
        print("strong correlation")

    # estimate strength of relation

    return r,p