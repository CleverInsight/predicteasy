# pylint: disable=R0904
"""
anomaly detection in given urls
"""
#import os
import requests
from bs4 import BeautifulSoup


class LinkGenuinityClassifier:
    """
    LinkGenuinityClassifier helps us to find the genuine score
    for the given `url`

    Usage:
    ======

        >>> clf = LinkGenuinityClassifier(url="https://economictime.com/articles.apx")
        >>> clf = LinkGenuinityClassifier(url="https://www.ndtv.com/india-news/
                  up-gangster-vikas-dubey-wanted-in-killing-of-8-cops-arrested
                  -in-ujjain-madhya-pradesh-2259611")
        >>> clf = LinkGenClassifier(url="[https://www.thehindu.com/sport/
                  cricket/sri-lankan-cricketer-kusal-mendis-arrested-for-
                  causing-fatal-motor-accident/article31993605.ece]
                  (https://www.thehindu.com/sport/cricket/sri-lankan-cricketer-
                  kusal-mendis-arrested-for-causing-fatal-motor-accident/
                  article31993605.ece)")
        >>> clf.score()


    Output:
    =======
        >>> {
                url: 'given_url',
                freshness: 0.8,
                geniuness: 0.8,
                spam_proximity: 0.4,
                is_date_relevant: True,
                is_nudity: False,
                is_violence: False,
                is_adult_content: False,
                is_content_bot_created: Fale,
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
        self.html_string = requests.get(url).text
        self.soup = BeautifulSoup(self.html_string, 'lxml')


    def get_ip_address(self):
        """
        pass
        """


    def get_title(self):
        """
        Fetching the title of the url given.

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url="https://www.ndtv.com/india-news/
                      up-gangster-vikas-dubey-wanted-in-killing-of-8-cops-arrested
                      -in-ujjain-madhya-pradesh-2259611")
            >>> clf.get_title()

        Output:
        =======

            >>> After 5-Day Run, UP Gangster Vikas Dubey Arrested At Madhya Pradesh Temple
        """
        return self.soup.title.text


    def get_content(self):
        """
        pass
        """


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


    def calculate_promises(self):
        """
        pass
        """


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


    def is_violence(self):
        """
        pass
        """


    def is_adult_content(self):
        """
        pass
        """


    def score_nudity(self):
        """
        pass
        """


    def score_violence(self):
        """
        pass
        """


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
