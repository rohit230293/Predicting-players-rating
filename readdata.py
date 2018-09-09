# Code to import the Dataset and performing preprocessing to it
import sqlite3
import pandas as pd

def readdata(dbname):
    # Create your connection.
    cnx = sqlite3.connect(dbname)
    df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
    
    '''
    All Columns:
    ['id', 'player_fifa_api_id', 'player_api_id', 'date', 'overall_rating',
           'potential', 'preferred_foot', 'attacking_work_rate',
           'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy',
           'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
           'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
           'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
           'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
           'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
           'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
           'gk_reflexes']
        
    '''
    # Dropping the columns ['id', 'player_fifa_api_id', 'player_api_id', \
    # 'date', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate']
    # As these fields are not not having significant data
    print("imported dataset >> ")
    print('-'*50)
    print(df.head(5))
    print('-'*50)
    print("Dropping the columns ['id', 'player_fifa_api_id', 'player_api_id', ")
    print("'date', 'preferred_foot', 'attacking_work_rate', 'defensive_work_rate'")
    print("As these fields are not not having significant data")
    print('-'*50)
    df.drop(['id', 'player_fifa_api_id', 'player_api_id', \
                  'date', \
                  'attacking_work_rate', \
                  'defensive_work_rate'], \
                    axis=1, inplace=True)
    
    # rows count before dropping NA
    rows_before_droppinig = df.shape[0]
    
    # Check null
    df.isnull().sum()
    
    # Dropping the nulls
    df = df.dropna()
    
    # rows count before dropping NA
    rows_after_droppinig = df.shape[0]
    
    # Check null again
    df.isnull().sum()
    
    # number of rows dropped
    diff = rows_before_droppinig - rows_after_droppinig
    print("Total number of rows dropped is {} wich is {:.4}% of total rows."   \
                  .format(diff,(diff/rows_before_droppinig)*100))
    print('-'*50)
    return df
