# pylint: disable=R0904
# pylint: disable=E1111
"""
anomaly detection in given urls
"""
#import os
import urllib
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


    def get_domain_name(self, remove_http=True):
        """
        Fetching the domain name of the url given.
        #22-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url= 'http://bit.ly/bcFOko')
            >>> clf.get_domain_name()

        Output:
        =======

            >>> bit.ly
        """
        uri = urllib.parse.urlparse(self.url)
        if remove_http:
            domain_name = f"{uri.netloc}"
        else:
            domain_name = f"{uri.netloc}://{uri.netloc}"
        return domain_name


    def get_title(self):
        """
        Fetching the title of the url given. #17-07-2020

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


    @staticmethod
    def check_relevance(content1, content2):
        """
        Checking if two given contents are related to each other or not.
        #20-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url="https://www.ndtv.com/india-news/
                      up-gangster-vikas-dubey-wanted-in-killing-of-8-cops-arrested
                      -in-ujjain-madhya-pradesh-2259611")
            >>> title = clf.get_title()
            >>> content = clf.get_content()
            >>> clf.check_relevance(title, content)

        Output:
        =======

            >>> True
        """
        documents = [content1, content2]
        count_vectorizer = CountVectorizer(stop_words='english')
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)
        value = cosine_similarity(sparse_matrix, sparse_matrix)[0, 1]
        return bool(True) if value >= 0.5 else bool(False)


    def parse_actual_date(self):
        """
        self.html_string parse and fetch the date of publications
        """


    def redirected_urls(self):
        """
        Returns all the redirected urls present in the website.
        #21-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url = "http://coreyms.com")
            >>> clf.count_redirected_urls()

        Output:
        =======

            >>> {'https://blog.codepen.io/radio/', 'http://carasantamaria.com/podcast/',
                 'https://twitter.com/CoreyMSchafer', 'https://coreyms.com/tag/standard-library',}
        """
        def is_valid(url):
            parsed = urllib.parse.urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)

        links = set()
        internal_urls = set()
        external_urls = set()
        urls = set()
        domain_name = urllib.parse.urlparse(self.url).netloc
        for a_tag in self.soup.find_all('a'):
            href = a_tag.attrs.get('href')
            if href == "" or href is None:
                continue
            href = urllib.parse.urljoin(self.url, href)
            parsed_href = urllib.parse.urlparse(href)
            href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
            if not is_valid(href):
                continue
            if href in internal_urls:
                continue
            if domain_name not in href:
                if href not in external_urls:
                    external_urls.add(href)
                    links.add(href)
                continue
            urls.add(href)
            links.add(href)
            internal_urls.add(href)
        return links


    def fetch_relevant_urls(self):
        """
        Returns all the relevant urls from the set of redirected urls.
        #23-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url = "http://coreyms.com")
            >>> clf.fetc_relevant_urls()

        Output:
        =======

            >>> {'https://twitter.com/CoreyMSchafer', 'https://coreyms.com/tag/standard-library',}
        """
        content = self.get_content()
        relevant_links = set()
        links = self.redirected_urls()
        for link in links:
            clf_1 = LinkGenuinityClassifier(link)
            if self.check_relevance(content, clf_1.get_content()) == bool(True):
                relevant_links.add(link)
        return relevant_links


    def fetch_irrelevant_urls(self):
        """
        Returns all the irrelevant urls from the set of redirected urls.
        #23-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url = "http://coreyms.com")
            >>> clf.fetc_irrelevant_urls()

        Output:
        =======

            >>> {'https://blog.codepen.io/radio/', 'http://carasantamaria.com/podcast/',}
        """
        links = self.redirected_urls()
        relevant_links = self.fetch_relevant_urls()
        return links - relevant_links


    def count_standard_url_shorteners(self):
        """
        Returns the number of standard url shorteners used.
        #22-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url = "http://bit.ly/bcFOko")
            >>> clf.count_standard_url_shorteners()

        Output:
        =======

            >>> 1
        """
        shorteners = ["bit.ly", "goo.gl", "Owl.ly", "Deck.ly", "Su.pr"
                      "lnk.co", "fur.ly", "moourl.com"]
        count = 0
        links = self.redirected_urls()
        links.add(self.url)
        for link in links:
            resp = urllib.request.urlopen(link)
            clf_1 = LinkGenuinityClassifier(url=link)
            clf_2 = LinkGenuinityClassifier(url=resp.url)
            given_url_domain = clf_1.get_domain_name()
            original_url_domain = clf_2.get_domain_name()
            if given_url_domain != original_url_domain:
                if given_url_domain in shorteners:
                    count = count + 1
        return count


    def count_unconventional_url_shorteners(self):
        """
        Returns the number of unconventional url shorteners used.
        #23-07-2020

        Usage:
        ======

            >>> clf = LinkGenuinityClassifier(url = "http://bit.ly/bcFOko")
            >>> clf.count_unconventional_url_shorteners()

        Output:
        =======

            >>> 0
        """
        shorteners = ["bit.ly", "goo.gl", "Owl.ly", "Deck.ly", "Su.pr"
                      "lnk.co", "fur.ly", "moourl.com"]
        count = 0
        links = self.redirected_urls()
        links.add(self.url)
        for link in links:
            resp = urllib.request.urlopen(link)
            clf_1 = LinkGenuinityClassifier(url=link)
            clf_2 = LinkGenuinityClassifier(url=resp.url)
            given_url_domain = clf_1.get_domain_name()
            original_url_domain = clf_2.get_domain_name()
            if given_url_domain != original_url_domain:
                if given_url_domain not in shorteners:
                    count = count + 1
        return count


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
