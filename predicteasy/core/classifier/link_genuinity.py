"""
Anomaly Detection for Websites
"""
#import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# pylint: disable=R0904
class LinkGenuinityClassifier:
    """
    LinkGenuinityClassifier helps us to find the genuine score
    for the given `url`

    Usage:
    ======

        >>> clf = LinkGenuinityClassifier(url="https://economictime.com/articles.apx")
        >>> clf = LinkGenClassifier(url="[https://www.thehindu.com/sport/cricket/sri-lankan-
        cricketer-kusal-mendis-arrested-for-causing-fatal-motor-accident/article31993605.ece]
        (https://www.thehindu.com/sport/cricket/sri-lankan-cricketer-kusal-mendis-arrested-for-
        causing-fatal-motor-accident/article31993605.ece)")
        >>> clf.score()


    Output:
    =======
        >>> {   url: 'given_url',
                freshness: 0.8,
                geniuness: 0.8,
                spam_proximity: 0.4,
                is_date_relevant: True,
                is_nudity: False,
                is_violence: False,
                is_adult_content: False,
                is_content_bot_created: False,
                total_irrelevant_out_link: 4,
                total_relevant_out_link: 10,
                total_geniun_in_link: 10,
                google_page_rank: 7,
                sentiment_distribution: [pos, pos, negative]
                sentiment_score: 0.66,
                topics: {
                    education: 10%,
                    sports: 70%,
                    media: 5%,
                    finance: 2%,
                    healthcare: 1%,
                    political: 20%
                }
         }
    """


    def __init__(self, url=""):
        self.url = url
        self.res = requests.get(url, stream=True)
        self.html_string = requests.get(url).text
        self.soup = BeautifulSoup(self.html_string, 'lxml')


    def get_ip_address(self):
        """
         get_ip_address is used for getting the IP adress of the URL provided.

        Usage:
        ======

            >>> clf = obj=LinkGenuinityClassifier(url="https://stackoverflow.com/questions/
            22492484/how-do-i-get-the-ip-address-from-a-http-request-using-the-requests-library")
            >>> print(clf.get_ip_address())


        Output:
        =======
            >>> 151.101.65.69
        """
        return self.res.raw._connection.sock.getpeername()[0]  # pylint: disable=W0212

    def get_title(self):
        """
        pass
        """


    def get_content(self):
        """
        get_content() is used for getting the content of the URL provided.

        Usage:
        ======

            >>> clf = obj=LinkGenuinityClassifier(url="https://stackoverflow.com/questions/
            22492484/how-do-i-get-the-ip-address-from-a-http-request-using-the-requests-library")
            >>> print(clf.get_ip_address())


        Output:
        =======
            >>> NEW DELHI: In a strong strategic signal to China amidst the military confrontation
            in eastern Ladakh, a US carrier strike group led by aircraft carrier USS Nimitz
            conducted a "cooperative" exercise with Indian warships in the Indian Ocean on Monday.
            The message was further amplified by a hard-nosed display of military intent by India,
            which has now deployed Jaguar maritime strike fighters in the Andaman and Nicobar
            archipelago that dominates China’s critical sea trade routes passing through the
            Malacca Strait.

        """
        contents = " "
        for content in self.soup.find_all('p'):
            contents = contents + content.text
        return contents



    def check_relevence(self, title, content):
        """
        pass
        """


    def parse_actual_date(self):
        """
        self.html_string parse and fetch the date of publications
        """


    def count_redirected_urls(self):
        """
        pass
        """


    def fetch_relevent_urls(self):
        """
        pass
        """


    def fetch_irrelevent_urls(self):
        """
        pass
        """


    def compare_content(self, content1, content2):
        """
        pass
        """


    def count_standard_url_shortners(self):
        """
        pass
        """


    def count_unconventional_url_shortners(self):
        """
        pass
        """


    def calculate_age_of_content(self, title, content):
        """
        pass
        """

    @classmethod
    def calculate_promises(cls):
        """
         calculate_promises is used for checking if the article contains any promise words.

        Usage:
        ======

            >>> clf = obj=LinkGenuinityClassifier(url="https://stackoverflow.com/questions/
            22492484/how-do-i-get-the-ip-address-from-a-http-request-using-the-requests-library")
            >>> print(clf.calculate_promises())


        Output:
        =======
            >>> False
        """
        sample_text = """Maharashtra, Mumbai, Pune Coronavirus News Live Updates:
        Maharashtra’s COVID-19 tally rose to 3,18,695 on Monday with the addition 
        of 8,240 new cases, while the death toll crossed the 12,000-mark, the state
        health department said. With 176 new deaths in a day, the state’s fatality 
        count increased to 12,030, the department said. A total of 5,460 patients were 
        discharged from hospitals, taking the number of recovered persons to 1,75,029, 
        the department said in a statement. There are 1,31,334 active cases in the state at present.
        Withthe tally of cases in Dharavi slum sprawl here has reached 2,492,the Brihanmumbai Municipal 
        Corporation (BMC) said. Dharavi now has 147 active cases.The BMC said 2,095 of 2,492 patients
        have already been discharged from hospitals.The number of COVID-19 cases in Pune district 
        crossed the 50,000-mark on Monday after 473 get get get get more people tested positive for the disease. 
        The district’s current case count stands at 51,885 with 1,343 fatalities.  Out of these 
        over 37,000 are reported from Pune city . A lockdown has been imposed in Pune city, Pimpri
        -Chinchwad and parts of Pune rural from July 14 to July 23, in an effort to curb the
        spread of the infection.
        Meanwhile, Maharashtra state textile minister Aslam Shaikh on Monday said he
        has tested positive for coronavirus and is isolating himself. He informed that 
        he is currently asymptomatic and has urged all his contacts to get themselves tested."""
        sample_text = sample_text.lower()
        file = open('C:/Users/HP/Desktop/Internship-CleverInsight/spam.txt', 'r')
        promise_words = [line.replace("\n", "").lower() for line in file.readlines()]
        count = 0
        target_text = []
        seperator = ' '
        for i in range(4):
            somegram = ngrams(sample_text.split(), i+1)
            for j in somegram:
                target_text.append(seperator.join(j))
        for i in target_text:
            if i in promise_words:
                count = count+1

        return count/len(target_text)


    def calculate_misleading_score(self):
        """
        pass
        """


    def calculate_block_sentiment(self):
        """
        pass
        """


    def fetch_topic_intensity(self):
        """
        pass
        """


    def is_nudity(self):
        """
        pass
        """

    @staticmethod
    def is_violence():
        """
        is_violence is used for checking if the article contains any violent text.

        Usage:
        ======

            >>> clf = obj=LinkGenuinityClassifier(url="https://stackoverflow.com/questions/
            22492484/how-do-i-get-the-ip-address-from-a-http-request-using-the-requests-library")
            >>> print(clf.is_violence())


        Output:
        =======
            >>> False
        """
        content_text = """Washington (CNN) - The billionaire NFL owner who serves as
        President Donald Trump's ambassador to the United Kingdom was investigated by the 
        State Department watchdog after allegations that he made racist and sexist comments to 
        staff and sought to use his government position to benefit the President's personal business
        in the UK, multiple sources told CNN."""
        word_tokens = word_tokenize(content_text)
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        violent_words = pd.read_csv('C:/Users/HP/Desktop/Internship-CleverInsight/violentword.csv',
                                    names=['words'])
        count = 0
        for i in filtered_sentence:
            if violent_words.isin([i]).any().any():
                count = count + 1
            else:
                continue
        return bool(True) if count > 0 else bool(False)

    def is_adult_content(self):
        """
        pass
        """


    def score_nudity(self):
        """
        pass
        """

    @staticmethod
    def score_violence():
        """
        score_violence is used for calculating hte amount of violent text the article contains.
        Usage:
        ======

            >>> clf = obj=LinkGenuinityClassifier(url="https://stackoverflow.com/questions/
            22492484/how-do-i-get-the-ip-address-from-a-http-request-using-the-requests-library")
            >>> print(clf.score_violence())


        Output:
        =======
            >>> 0.5
        """
        content_text = """Washington (CNN) - The billionaire NFL owner who serves
        as President Donald Trump's ambassador to the United Kingdom was investigated by the 
        State Department watchdog after allegations that he made racist and sexist comments to 
        staff and sought to use his government position to benefit the President's personal business 
        in the UK, multiple sources acid"""
        word_tokens = word_tokenize(content_text)
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        violent_words = pd.read_csv('C:/Users/HP/Desktop/Internship-CleverInsight/violentword.csv',
                                    names=['words'])
        count = 0
        for i in filtered_sentence:
            if violent_words.isin([i]).any().any():
                count = count + 1
            else:
                continue
        return count/len(filtered_sentence)

    def score_adult_content(self):
        """
        pass
        """


    def calculate_content_freshness(self):
        """
        pass
        """


    def get_pagerank_score(self, url):
        """
        pass
        """


    def learn(self):
        """

        pass

        """


    def score(self):
        """
        pass
        """
