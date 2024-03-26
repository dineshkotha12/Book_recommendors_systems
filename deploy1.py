#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


# data loading
final=  pd.read_csv(r'C:\Users\study\Downloads\csvfiles\output_file.csv')
recommendations = final.sort_values(by='Book_Rating', ascending=False).head(5)


# Convert 'User-ID' and 'Book-Rating' to numeric types if they are not
final['User_ID'] = pd.to_numeric(final['User_ID'], errors='coerce')
final['Book_Rating'] = pd.to_numeric(final['Book_Rating'], errors='coerce')

# Drop NaN values if necessary
final = final.dropna(subset=['User_ID', 'Book_Rating'])

# Create the pivot table
pt = final.pivot_table(index='Book_Title', columns='User_ID', values='Book_Rating', fill_value=0)




# In[6]:


pt


# In[8]:


pt = final.pivot_table(index='Book_Title',columns= "User_ID" ,values='Book_Rating', fill_value=0)
pt.fillna(0, inplace=True)

def recommend(Book_name):
    index = np.where(pt.index == Book_name)[0][0] 
    similarity_scores = cosine_similarity(pt)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key= lambda x: x[1], reverse = True)[1:6]
    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books




def main():
    st.title("Book Recommendation System")
    st.header("User-Based recommender's system")
     

    book_name = st.text_input("Enter book name  or Book Title Here") #user based

    if st.button("Recommend"):
        try:
            book_name = int(book_name)
            if book_name in pt.columns:
                user_ratings= pt[book_name].dropna()
                top_rating_books= user_ratings.sort_values(ascending=False).head(10)
                
                st.subheader(f"top 5 rated books for User {book_name}:")
                st.write(top_rating_books)
            else:
                st.warning("invalid user id")
        except ValueError:
             if book_name in pt.index:
                 recommended_books = recommend(book_name)
                 st.subheader(f"books similar to '{book_name}':")
                 st.write(recommended_books)
                 
             else:
                 st.warning(f"not found")

    st.sidebar.title("user based")
    book_input = st.text_input("enter book name")
    
    if st.button("recommend similar books"):
         if book_input in pt.index:
             recommended_books = recommend(book_input)
             st. subheader(f"similar books are")
             st.write(recommended_books)
         else:
             st.warning("not found")

        

if __name__ == "__main__":
    main()





# In[ ]:




