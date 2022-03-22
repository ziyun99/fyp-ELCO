import pandas as pd

# load df from csv file
an_df = pd.read_csv("data/data_attribute_ground_truth.csv")
print(an_df.head())
col_names = an_df.columns.values
print(col_names)
an_dict = dict( zip(an_df["concept"], an_df["attribute"]))
print(an_dict)
# def google_definition(user_input):
#     import requests 
#     import bs4 
#     searchname = user_input.replace(' ', '+')
#     url = "https://google.com/search?q=define+" + searchname
#     request_result = requests.get( url )
#     soup = bs4.BeautifulSoup( request_result.text, "html.parser" )
#     heading_object=soup.find_all( "div" , class_='BNeawe s3v9rd AP7Wnd' )
#     definition = heading_object[2].getText()
#     return definition

# query = "fresh bread"
# definition = google_definition(query)
# print(definition)