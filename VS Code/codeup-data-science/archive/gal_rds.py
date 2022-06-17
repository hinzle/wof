import pprint

def rec_dig_sum(n):
    '''
    Returns the recursive digit sum of an integer.

    Parameter
    ---------
    n: int

    n=int(n)
    y=int()

    for x in range(n+1):
        y=y+x
    '''
    strr = str(n)
    list_of_number = list(map(int, strr.strip()))
    

    '''
    Returns
    -------
    rec_dig_sum: int
       the recursive digit sum of the input n
    
    return y
    '''
    return sum(list_of_number)

def distr_of_rec_digit_sums(low: int, high: int):
    '''
    Returns a dictionary representing the counts
    of recursive digit sums within a given range.

    Parameters
    ----------
    low: int
        an integer, 0 or positive, representing
        the lowest value in the range of integers
        for which finding the recursive digit sum
    high: int
        a positive integer greater than low, the
        inclusive upper bound for which finding
        the recursive digit sum
    '''
    rds_lst=[]
    for x in range(high+1):
        rds_lst.append(x)
    rds_lst[0:low]=[]
    print(rds_lst)    
    
    dict_rds={}
    for x in rds_lst:
        print(rds_lst[x])
        
        rds_temp=rec_dig_sum(rds_lst[x])
        dict_rds[rds_temp]=x
        
    pprint.pprint(dict_rds)

    '''
    Returns
    -------
    dict_of_rec_dig_sums: {int:int}
        returns a dictionary where the keys are
        the recursive digit sums and the values
        are the counts of those digit sums occurring
    '''
    return dict_rds

#n=input("Pick an integer: ")
#rds=rec_dig_sum(n)
#print("Recursive digit sum:",rds)

low=input("Lowest integer in sequence (0:inf):")
low=int(low)
if low < 0:
   print("Lowest integer must be zero or greater.")
high=input("Highest integer in sequence (O:inf):")
high=int(high)
if high < 0:
   high=input("Highest integer must be zero or greater.")
distr_rds=distr_of_rec_digit_sums(low, high)
print("Distribution of RDS values:",distr_rds)