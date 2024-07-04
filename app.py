import helper
import re
import pandas as pd
from wordcloud import WordCloud
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sn
import emoji

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_message' : messages, 'date' : dates})
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%y, %H:%M - ")
    user = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if(entry[1:]):
            user.append(entry[1])
            messages.append(entry[2])
        else:
            user.append('Group Notification')
            messages.append(message)
    df['user'] = user
    df['message'] = messages
    df = df.drop(['user_message'], axis = 1)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    return df

def sidenames(df):
    names = df.user.unique()
    names = names[names != 'Group Notification']
    names = np.sort(names)
    names = np.append(['Overall'], names)
    return names

def most_busy_day(dfe):
    monname = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    mesonday = [len(dfe[dfe['day_name'] == i]) for i in monname]
    
    mon_mes = [[monname[i], mesonday[i]] for i in range(len(monname))]
    sorted_mon_mes = sorted(mon_mes, key=lambda x: x[1], reverse=True)
    return sorted_mon_mes

def most_busy_month(dfe):
    dayname = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', ' September', 'October', 'November', 'December']
    mesonday = [len(dfe[dfe['month_name'] == i]) for i in dayname]
    
    day_mes = [[dayname[i], mesonday[i]] for i in range(len(dayname))]
    sorted_day_mes = sorted(day_mes, key=lambda x: x[1], reverse=True)
    print(sorted_day_mes)
    return [sorted_day_mes, day_mes]

def extract_unique_emojis(dfe):
    unique_emojis = set()
    emojis_text = ' '.join(dfe['message'].astype(str))
    kk = emojis_text
    
    # Extract emojis from the text using emoji library
    emojis = ''.join(c for c in emojis_text if c in emoji.EMOJI_DATA)
    
    # Convert emojis into their text representation
    emojis_text = emoji.demojize(emojis)
    
    # Count the occurrence of each unique emoji
    for emoji_char in emojis:
        unique_emojis.add(emoji_char)
    print(unique_emojis)
    
    # Create a dictionary to store counts of each emoji
    emoji_counts = {}
    for emoji_char in unique_emojis:
        for emoji_word in kk:
            if(emoji_char == emoji_word):
        # emoji_counts[emoji_char] = emojis_text.count(emoji_char)
                if(emoji_char in emoji_counts.keys()):
                    emoji_counts[emoji_char] = emoji_counts[emoji_char] + 1
                else:
                    emoji_counts[emoji_char] = 1
    sorted_emoji_counts = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_emoji_counts)
    words = [word for word, _ in sorted_emoji_counts]
    frequencies = [freq for _, freq in sorted_emoji_counts]
    top_words = words[:5]
    top_frequencies = frequencies[:5]
    fig, ax = plt.subplots()
    ax.bar(top_words, top_frequencies, color = 'red')
    ax.set_xlabel('Emojis')
    ax.set_ylabel('Frequency')
    plt.title("Top Emojis")
    st.pyplot(fig)
    return sorted_emoji_counts


st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocess(data)
    names = sidenames(df)
    name = st.sidebar.selectbox("Choose the user", names)
    if(st.sidebar.button('Show Analysis')):
        st.markdown("<h1 style='border-bottom : 2px solid white; text-align: center;'>Top Statistics</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # st.header("Total Messages")
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Total Messages</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = helper.fetch_stats(name, df)[0]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        with col2:
            # st.header("Total Words")
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Total Words</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = helper.fetch_stats(name, df)[1]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            # st.header("Medias Shared")
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Medias Shared</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = helper.fetch_stats(name, df)[2]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        with col2:
            # st.header("Links Shared")
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Links Shared</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = helper.fetch_stats(name, df)[3]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>All Messages</h1>", unsafe_allow_html=True)
        dfe = helper.showdf(name, df)
        st.dataframe(dfe)

        dfe['month_year'] = dfe.apply(helper.combine_month_year, axis=1)
        dfe['day_month_year'] = dfe.apply(helper.combine_day_month_year, axis=1)
        all_months = helper.list_months(dfe)
        all_days = helper.list_days(dfe)
        print(all_days)
        print(all_days)
        # timeline = st.selectbox("Choose the timeslot", ['Daily Timeline', 'Monthly Timeline', 'Yearly Timeline'])
        day_mes = most_busy_day(dfe)
        mon_mes = most_busy_month(dfe)[0]
        monmes = most_busy_month(dfe)[1]

        st.title("Daily Timeline")
        helper.daily_timeline(dfe, all_months, all_days)

        st.title("Monthly Timeline")
        helper.monthly_timeline(dfe, monmes)
        st.title("Message Counts")
        col1, col2 = st.columns(2)
        with col1:
            helper.most_busy_day_graph(dfe, day_mes)
        with col2:
            helper.most_busy_month_graph(dfe, mon_mes)
        helper.day_time_graph(dfe)
        st.title("Top Users")
        col1, col2 = st.columns(2)
        with col1:
            user_messages = helper.most_busy_users(name, df, dfe)
        with col2:
            helper.user_percentages(name, user_messages)
        st.title("Word Cloud")
        helper.generate_wordcloud(df)
        st.title("Top Words")
        sorted_word_count = helper.mostcommonwords(dfe)
        st.title("Emoji Counts")
        col1, col2 = st.columns(2)
        with col1:
            emoji_count = extract_unique_emojis(dfe)
        with col2:
            dfes = pd.DataFrame(emoji_count, columns=['Emojis', 'Frequency'])
            st.dataframe(dfes.head())