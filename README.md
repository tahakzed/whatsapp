What my WhatsApp chats revealed: A data-driven journey
As an aspiring data analyst, I'm fortunate to have access to the tools necessary for performing meaningful analysis. Yet, these tools are only valuable when actively used. I delved into my creative thoughts, eager to employ these tools in a way that was both fun and personally fulfilling✌️. Therefore, rather than presenting this article as a definitive guide on best practices in NLP, I invite you to view it as a personal narrative of my own explorations and the approaches I employed in my project. This is a glimpse into my journey, not a blueprint for others to follow.
Well, one of the things that's deeply personal to me is the conversations I have with people, and the digital counterpart to this is WhatsApp (and iMessage, but that's another story). I was thrilled to discover I could download my WhatsApp chats. Of course, I had consent - well, from just one person, actually. But let's be clear: I wasn't plotting any grand heists via WhatsApp. Still, to keep things light, I swapped real names for fictional ones. So, no lawsuits, please!
In this article, I'll guide you through a little pet project of mine where I downloaded my WhatsApp chats and conducted some basic analysis. Along the way, I'll highlight the mistakes I made and share the thought process behind each analysis. For those interested in diving deeper, I've included a GitHub link to the complete code at the end of this article, along with key code snippets where relevant. 
Let's dive in!
Dataframe generation
I didn't download all my WhatsApp chats because I wanted to download only those conversations I had in English. However, I ended up downloading some German messages as well, since many of my friends are bilingual or even trilingual.
I won't explain the steps one should take to download WhatsApp chats. A simple Google search will do the trick!
The generated file was a txt file which contained three pieces of information: the timestamp, sender of the message and the message. the data was in the format: <date>, <time> - <sender>: <message>. 
Format of generated fileBecause there is a pattern, the most sensible way to extract each individual message would be to use regular expressions. Simple string splitting might end up complicating things. Trust me, I tried that first. I used the regular expression of the following form: '(.*?) - (.*?): (.*)' which extracted all three pieces of information. 
    with open(file_path, 'r', encoding = 'utf8') as file:
        for line in file:
            sender_message_match = re.match(r'(.*?) - (.*?): (.*)', line)
            if sender_message_match:
                timestamp, sender, message = sender_message_match.groups()
                senders.append(sender.strip())
                messages.append(message.strip())  
                split_timestamp = timestamp.split(', ')
                date.append(split_timestamp[0].strip())
                time.append(split_timestamp[1].strip())
Only thing left afterwards was to split the timestamp using the ',' separator. Here, string splitting is a better idea. After I had these four pieces of information I ended up creating a dataframe out of it for future.
Generated dataframeNow that the dataframe is generated, we can move on to the cool stuff!
Statistics
Text messages and media messages count
I wanted to find out who sends me the most text messages and who sends the most media. Since I download the data without media so whenever a media (photo, audio, video etc) was sent or received, WhatsApp simply labeled the content of the message as '<Media omitted>'.
I counted the number of text messages and the number of media received by simple filtering and counting of the rows.
Number of text messages receivedWord frequency
Just calculating the frequency was not enough. I wanted to visualize it. I deemed word cloud as a good option.
I performed text cleaning before doing that. I will explain how in the later section. 
Word cloud for received text messagesMessage length
In this step, I constructed a new dataframe based on the original, focusing on two key columns: 'Senders' and 'Message Length.' I grouped the data by sender to aggregate the messages, and then calculated the average length of the messages for each sender. This produced a series which I subsequently transformed back into a dataframe. This adjustment made it easier to visualize the data through a bar graph.
Comparison of average message lengths among difference peopleGender ratios
The data did not come pre-labeled with gender information, so I assigned the labels myself through a simple mapping process. Since I am certain of the genders of the individuals I communicate with, please don't throw any allegations 🤞.
Gender ratios Active hours analysis
Since we all are busy in our lives. There are only a certain number of hours where we get time to chat with our friends and families. I never pay attention to the time when I receive a text from someone. However, I can find that out using the data I have. Doing it for all the texts I receive from everyone as one large group is not that interesting and there may not be much difference between different periods of time. However, doing this for an individual is interesting and much more meaningful.
So, the main idea was to find out the frequency of messages in different time periods throughout 24 hours. My approach to this involved firstly to divide 24 hours into 6 time periods having 4 hours each and then assigning each row a label of that time period based on the time of the message. Instead of the content of the message I simply put a constant 1 to make frequency calculation easier. 
A glimpse of the resulting dataframeAll that was left later was to group the data based on the time period and sum up all rows in the message columns. This gave me the frequency of messages in different periods of time throughout 24 hours.
from datetime import datetime

def group_time(time):
    intervals = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
    time_intervals = [datetime.strptime(time, '%H:%M') for time in intervals]
    groups = ['00-04', '04-08', '08-12', '12-16', '16-20', '20-00']
    
    if time.hour >= time_intervals[0].hour and time.hour<time_intervals[1].hour:
        return groups[0]
    elif time.hour >= time_intervals[1].hour and time.hour<time_intervals[2].hour:
        return groups[1]
    elif time.hour >= time_intervals[2].hour and time.hour<time_intervals[3].hour:
        return groups[2]
    elif time.hour >= time_intervals[3].hour and time.hour<time_intervals[4].hour:
        return groups[3]
    elif time.hour >= time_intervals[4].hour and time.hour<time_intervals[5].hour:
        return groups[4]
    elif time.hour >= time_intervals[5].hour and time.hour<time_intervals[0].hour:
        return groups[5]
The plot of active hoursConversation initiation
What's the frequency of first messages a person sends in a two-person conversation? Or in other words, who's usually kicking things off? This analysis can shed light on the dynamics of a relationship, though I wouldn't use it to judge the quality of that relationship.
The data wasn't labeled with which messages started a conversation, so I had to figure that out myself. I consider a message to be the start of a new conversation if it's sent or received after a day or more since the last one. This rule generally works well, unless you're in a relationship - then it might not be the best assumption. If the gap between messages is a day or longer, I mark it as a new message. I labeled each conversation-starter with the person's name, and used 'in_convo' for ongoing messages, because let's face it - 'null' or 'None' just doesn't look good.
A glimpse of the resulting

A simple tally of names in the 'convo_starter' column then answered my earlier questions about who initiates conversations more frequently.

Since these are texts, they carry a certain sentiment/emotion with them.
You guessed it.
Sentiment analysis
At the time, I had two approaches in mind, specifically two types of algorithms. The first was a machine learning algorithm. This approach consumes less power, but it requires labeled data to train the model. Since these were real conversations, they weren't labeled, and I neither had the time nor the energy to label each text myself. We're talking about thousands of rows here. The second approach was using deep learning. The unavailability of labeled data was still an issue here as well. However, HuggingFace kindly developed some pre-trained language models suitable for sentiment analysis. I opted for this approach, of course. Having previously seen one of these pre-trained models, namely BERT, in action, I was eager to get started with installing sklearn and other necessary libraries. Since BERT is a hefty model, it makes sense to use a GPU for faster computations. However, my laptop quickly let me down by confirming that BERT indeed IS a heavy model.
I then searched for lighter pre-trained models and discovered AlBERT, which is essentially a lighter version of BERT. But before diving into sentiment prediction, I needed to clean the texts. This involved removing punctuation, stopwords, some extraneous words, and lemmatizing. It was only later that I learned stemming could have been an option as well, but by then I was too lazy to rerun the model.
# text cleaning
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def generate_clean_text(text_list):
    
    all_text = ' '.join(line for line in text_list if line not in ['<Media omitted>'] and type(line)==str)
    eg_stopwords = set(stopwords.words('english'))
    garbage = ['s', 't', 'n\'t', '\'s', '\'ll']
    
    tokens = word_tokenize(all_text)
    cleaned_text = ' '.join(word.lower() for word in tokens if word not in eg_stopwords and word not in garbage)
    return cleaned_text
AlBERT delivered the sentiments of my texts. Most were positive, which was kind of disappointing if you ask me. But even more disappointing was that there were only two negatives, and both of them were emails for some reason. It seems AlBERT isn't a fan of professional communication forms.
Finally, I added the sentiment labels to my original dataframe.
Sentiment DistributionSentiment distribution on weekdays vs weekends
By using a simple function and tapping into the datetime library, I crafted a dataframe that tracked the counts of messages sent and received on weekends and weekdays. I then analyzed the number of messages associated with each sentiment for both time periods. Next, I calculated the percentage difference between positive and neutral messages received on weekdays versus weekends. 

Interestingly, I found that I receive more positive than neutral messages on weekends compared to weekdays. Although the total number of positive messages is higher on weekdays - simply because I get more messages overall - the difference swings more noticeably in favor of positives on weekends. And that makes sense, right? Because, well, weekends.
Conclusion
As we wrap up this exploration of my WhatsApp data, I hope you've found the journey as enlightening and entertaining as I did. Analyzing personal communication to uncover patterns and sentiments has revealed not just how much data we generate in casual conversations, but also how much insight lies hidden within them.
While the technical aspects of this project required a blend of machine learning and natural language processing, the true joy came from the surprises hidden in the data - those unexpected findings that challenge our perceptions of everyday interactions.
For those interested in delving deeper or replicating this analysis with your own data, I've shared all the code and methodologies on my GitHub. Feel free to fork, star, or contribute to the repository:
[Insert GitHub link here]
Whether you're a data enthusiast, a curious reader, or somewhere in between, I encourage you to consider the stories your own data might tell. Thank you for joining me on this data-driven adventure. Until next time, keep exploring and questioning the data that surrounds us every day. :)
