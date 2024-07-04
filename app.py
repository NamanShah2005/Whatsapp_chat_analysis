import re
import pandas as pd
from wordcloud import WordCloud
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sn
import emoji
from urlextract import URLExtract
import re
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sn
from wordcloud import WordCloud
# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import pipeline

# model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
# tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

extractor = URLExtract()
# nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# def preprocess(data):
#     pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s-\s'
#     messages = re.split(pattern, data)[1:]
#     dates = re.findall(pattern, data)
#     df = pd.DataFrame({'user_message' : messages, 'date' : dates})
#     df['date'] = pd.to_datetime(df['date'], format="%d/%m/%y, %H:%M - ")
#     user = []
#     messages = []
#     for message in df['user_message']:
#         entry = re.split('([\w\W]+?):\s', message)
#         if(entry[1:]):
#             user.append(entry[1])
#             messages.append(entry[2])
#         else:
#             user.append('Group Notification')
#             messages.append(message)
#     df['user'] = user
#     df['message'] = messages
#     df = df.drop(['user_message'], axis = 1)
#     df['year'] = df['date'].dt.year
#     df['month'] = df['date'].dt.month
#     df['month_name'] = df['date'].dt.month_name()
#     df['day'] = df['date'].dt.day
#     df['day_name'] = df['date'].dt.day_name()
#     df['hour'] = df['date'].dt.hour
#     df['minute'] = df['date'].dt.minute
#     return df

def sidenames(df):
    names = df.user.unique()
    names = names[names != 'Group Notification']
    names = np.sort(names)
    names = np.append(['Overall'], names)
    return names

def fetch_stats(name, df):
    if(name == 'Overall'):
        words = []
        urls = []
        for message in df['message']:
            words.extend(message.split())
            urls.extend(extractor.find_urls(message))
        return df.shape[0], len(words), df[df['message'] == '<Media omitted>\n'].shape[0], len(urls)
    

    words = []
    urls = []
    for message in df[df['user'] == name]['message']:
            words.extend(message.split())
            urls.extend(extractor.find_urls(message))


    return df[df['user'] == name].shape[0], len(words), df[(df['message'] == '<Media omitted>\n') & (df['user'] == name)].shape[0], len(urls)

def showdf(name, df):
     if(name == 'Overall'):
          return df
     return df[df['user'] == name]

def list_months(dfe):
    months = dfe['month_year']
    months_dates = [datetime.strptime(month, "%B-%Y") for month in months]
    first_month = min(months_dates)
    last_month = max(months_dates)
    all_months = []
    current_month = first_month
    while current_month <= last_month + timedelta(days=30):
        all_months.append(current_month.strftime("%B-%Y"))
        current_month += timedelta(days=30)
    return all_months

def list_days(dfe):
    days = dfe['day_month_year'].unique()
    day_dates = [datetime.strptime(day, "%d-%B-%Y") for day in days]
    first_day = min(day_dates)
    last_day = max(day_dates)
    all_days = []
    current_day = first_day
    while current_day <= last_day:
        all_days.append(current_day.strftime("%d-%B-%Y"))
        current_day += timedelta(days=1)
    return all_days

def combine_month_year(row):
    return f"{row['month_name']}-{row['year']}"

def combine_day_month_year(row):
    return f"{row['day']}-{row['month_name']}-{row['year']}"

def monthly_timeline(dfe, day_mes):
    days = [day_mes[i][0] for i in range(len(day_mes))]
    mes = [day_mes[i][1] for i in range(len(day_mes))]
    setter = dfe['month_year'].unique()
    data = pd.DataFrame({
        'x': days,
        'y': mes
    })
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'], color='green')
    ax.set_xticklabels(days, rotation=90)
    ax.set_xlabel('Months')
    ax.set_ylabel('Message Count')
    st.pyplot(fig)

def daily_timeline(dfe, all_months, all_days):
    for i in range(len(all_days)):
        all_days[i] = re.sub(r'\b0(\d)', r'\1', all_days[i])
    pattern = '\d{1,2}-'
    first_date_first = re.findall(pattern, all_days[0])
    first_date = first_date_first[0][:-1]
    dfe['day_month_year'] = dfe['day_month_year'].str.replace(r'\b0\d', r'\b0(\d)')
    data = pd.DataFrame({
        'x': all_days,
        'y': [len(dfe[dfe['day_month_year'] == i]) for i in all_days]
    })
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'], color='red')
    first_days_of_months = [f'{first_date}-{month}' for month in all_months]
    ax.set_xticks(first_days_of_months)
    ax.set_xticklabels(first_days_of_months, rotation=90)
    ax.set_xlabel('Days')
    ax.set_ylabel('Message Count')
    st.pyplot(fig)

def most_busy_day_graph(dfe, day_mes):
    days = [day_mes[i][0] for i in range(len(day_mes))]
    mes = [day_mes[i][1] for i in range(len(day_mes))]
    setter = dfe['month_year'].unique()
    data = pd.DataFrame({
        'x': days,
        'y': mes
    })
    fig, ax = plt.subplots()
    ax.bar(data['x'], data['y'], color='purple')
    ax.set_xticklabels(days, rotation=90)
    ax.set_xlabel('Days')
    ax.set_ylabel('Message Count')
    # ax.title("Day wise message count")
    st.pyplot(fig)

def most_busy_month_graph(dfe, day_mes):
    days = [day_mes[i][0] for i in range(len(day_mes))]
    mes = [day_mes[i][1] for i in range(len(day_mes))]
    setter = dfe['month_year'].unique()
    data = pd.DataFrame({
        'x': days,
        'y': mes
    })
    fig, ax = plt.subplots()
    ax.bar(data['x'], data['y'], color='green')
    ax.set_xticklabels(days, rotation=90)
    ax.set_xlabel('Months')
    ax.set_ylabel('Message Count')
    st.pyplot(fig)

def day_time_graph(dfe):
    dayname = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    hours = list(range(24))
    column_labels = ['{}-{}'.format(hour, hour + 1) for hour in hours]
    day_time = [[len(dfe[(dfe['day_name'] == day) & (dfe['hour'] == hour)]) for hour in hours] for day in dayname]
    df_day_time = pd.DataFrame(day_time, columns=column_labels, index=dayname)
    fig, ax = plt.subplots()
    sn.heatmap(df_day_time, annot=False)
    plt.xlabel("period")
    plt.ylabel("day name")
    st.pyplot(fig)

def most_busy_users(name, df, dfe):
    if(name == 'Overall'):
        names = df['user'].unique().tolist()
        chats = [len(df[df['user'] == names[i]]) for i in range(len(names))]
        names_chats = [[names[i], chats[i]] for i in range(len(names))]
        sorted_names_chats = sorted(names_chats, key=lambda x: x[1], reverse=True)
        top_ten_users = sorted_names_chats[0:5]
        top_names = []
        top_messages = []
        for i in range(len(top_ten_users)):
            top_names.append(top_ten_users[i][0])
            top_messages.append(top_ten_users[i][1])
        fig, ax = plt.subplots()
        ax.bar(top_names, top_messages, color = 'red')
        ax.set_xlabel('Username')
        ax.set_xticklabels(top_names, rotation=90)
        ax.set_ylabel('Message Count')
        plt.title("Top Users")
        st.pyplot(fig)
        return sorted_names_chats
    else:
        return None
    
def user_percentages(name, user_mes):
    if(name == 'Overall'):
        user = []
        mes = []
        sum_mes = 0
        for i in range(len(user_mes)):
            user.append(user_mes[i][0])
            mes.append(user_mes[i][1])
            sum_mes = sum_mes + user_mes[i][1]
        mes_percent = []
        for i in range(len(mes)):
            percentage = (mes[i]/sum_mes)*100
            mes_percent.append(percentage)
        dfs = pd.DataFrame(mes_percent, user, columns=['percentages'])
        st.dataframe(dfs)

def generate_wordcloud(dfe):
    dfe = dfe[dfe['message'] != '<Media omitted>\n']
    dfe = dfe[dfe['message'] != 'This message was deleted\n']
    dfe = dfe[dfe['message'] != 'null\n']
    dfe = dfe[dfe['message'] != 'You deleted this message\n']
    text = ' '.join(dfe['message'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(fig)

def mostcommonwords(dfe):
    dfe = dfe[dfe['message'] != '<Media omitted>\n']
    dfe = dfe[dfe['message'] != 'This message was deleted\n']
    dfe = dfe[dfe['message'] != 'null\n']
    dfe = dfe[dfe['message'] != 'You deleted this message\n']
    text = ' '.join(dfe['message'].astype(str).tolist())
    words = text.split()
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    words = [word for word, _ in sorted_word_count]
    frequencies = [freq for _, freq in sorted_word_count]
    top_words = words[:10]
    top_frequencies = frequencies[:10]
    fig, ax = plt.subplots()
    ax.barh(top_words, top_frequencies, color = 'brown')
    ax.set_xlabel('Top Words')
    ax.set_ylabel('Frequency')
    plt.title("Top Words")
    st.pyplot(fig)
    return sorted_word_count

# def sentiment_analysis(dfe):
#     text = ' '.join(dfe['message'].astype(str))
#     print(text)
#     result = nlp(text)
#     print(result)
    # st.header("Sentiment Analysis")
    # st.write(f"Label : {nlp(result)['label']}")
    # st.write(f"Score : {nlp(result)['score']}")

def preprocess(data):
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s-\s'
    pattern2 = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s[a,p]m\s-\s'
    pattern3 = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}'
    # pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s[a,p]m\s-\s'
    # messages = re.split(pattern, data)[1:]
    # dates = re.findall(pattern, data)

    if re.findall(pattern, data):
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        date_format = "%d/%m/%y, %H:%M - "
    else:
        messages = re.split(pattern2, data)[1:]
        dates = re.findall(pattern2, data)
        date_format = "%d/%m/%y, %I:%M %p - "
    # df = pd.DataFrame({'user_message' : messages, 'date' : dates})

    df = pd.DataFrame({'user_message' : messages, 'date' : dates})
    df['date'] = pd.to_datetime(df['date'], format=date_format)
    # df['date'] = pd.to_datetime(df['date'], format="%d/%m/%y, %H:%M - ")
    # df['date'] = pd.to_datetime(df['date'], format="%d/%m/%y, %H:%M pm - ")
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
    return [sorted_day_mes, day_mes]

def extract_unique_emojis(dfe):
    unique_emojis = set()
    emojis_text = ' '.join(dfe['message'].astype(str))
    kk = emojis_text
    emojis = ''.join(c for c in emojis_text if c in emoji.EMOJI_DATA)
    emojis_text = emoji.demojize(emojis)
    for emoji_char in emojis:
        unique_emojis.add(emoji_char)
    emoji_counts = {}
    for emoji_char in unique_emojis:
        for emoji_word in kk:
            if(emoji_char == emoji_word):
                if(emoji_char in emoji_counts.keys()):
                    emoji_counts[emoji_char] = emoji_counts[emoji_char] + 1
                else:
                    emoji_counts[emoji_char] = 1
    sorted_emoji_counts = sorted(emoji_counts.items(), key=lambda x: x[1], reverse=True)
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
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Total Messages</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = fetch_stats(name, df)[0]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        with col2:
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Total Words</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = fetch_stats(name, df)[1]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Medias Shared</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = fetch_stats(name, df)[2]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        with col2:
            styled_stats = f"<h1 style='font-size:35px; text-align:center; justify-content: center'>Links Shared</h1>"
            st.markdown(styled_stats, unsafe_allow_html=True)
            stats = fetch_stats(name, df)[3]
            styled_stats = f"<h2 style='color: blue; text-align:center; justify-content: center'>{stats}</h2>"
            st.markdown(styled_stats, unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>All Messages</h1>", unsafe_allow_html=True)
        dfe = showdf(name, df)
        st.dataframe(dfe)
        dfe['month_year'] = dfe.apply(combine_month_year, axis=1)
        dfe['day_month_year'] = dfe.apply(combine_day_month_year, axis=1)
        all_months = list_months(dfe)
        all_days = list_days(dfe)
        day_mes = most_busy_day(dfe)
        mon_mes = most_busy_month(dfe)[0]
        monmes = most_busy_month(dfe)[1]
        st.title("Daily Timeline")
        daily_timeline(dfe, all_months, all_days)
        st.title("Monthly Timeline")
        monthly_timeline(dfe, monmes)
        st.title("Message Counts")
        col1, col2 = st.columns(2)
        with col1:
            most_busy_day_graph(dfe, day_mes)
        with col2:
            most_busy_month_graph(dfe, mon_mes)
        day_time_graph(dfe)
        st.title("Top Users")
        col1, col2 = st.columns(2)
        with col1:
            user_messages = most_busy_users(name, df, dfe)
        with col2:
            user_percentages(name, user_messages)
        st.title("Word Cloud")
        generate_wordcloud(df)
        st.title("Top Words")
        sorted_word_count = mostcommonwords(dfe)
        st.title("Emoji Counts")
        col1, col2 = st.columns(2)
        with col1:
            emoji_count = extract_unique_emojis(dfe)
        with col2:
            dfes = pd.DataFrame(emoji_count, columns=['Emojis', 'Frequency'])
            st.dataframe(dfes.head())
        # st.header("Sentiment Analysis")
        # sentiment_analysis(dfe)