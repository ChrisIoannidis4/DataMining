import pandas as pd

def map_to_binary(x):
    """
    Basically a step function
    """
    if x>0:
        return 1
    else:
        return 0
    
def create_csv(data, fname):
    """
    Create csv file, keeping only the 41 specified independent variables and the label column  

    :data: original eclipse data as pandas dataframe
    :fname: csv filename  
    """

    for col in data.columns:
        if col.startswith(('FOUT', 'MLOC', 'NBD', 'PAR', 'VG', 'NOF', 'NOM', 'NSF', 'NSM', 'ACD', 'NOI', 'NOT', 'TLOC', 'NOCU', 'pre', 'post')):
            continue
        else:
            data.drop(columns=[col], inplace=True)

    # change column order so that label column is last
    temp_cols =data.columns.tolist()
    new_cols=[temp_cols[0]] + temp_cols[2:] + [temp_cols[1]]
    data= data[new_cols]

    # make post column binary 
    post_col = data['post'].apply(map_to_binary)
    post_col.apply(map_to_binary)
    data['post'] = post_col

    # save to csv
    data.to_csv(f'./{fname}.csv', sep=',', index=False)

    return

data = pd.read_csv('./eclipse-metrics-packages-2.0.csv', sep=';')
test_data = pd.read_csv('./eclipse-metrics-packages-3.0.csv', sep=';')
create_csv(data, 'eclipse_train')
create_csv(test_data, 'eclipse_test')