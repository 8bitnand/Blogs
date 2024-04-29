from bs4 import BeautifulSoup
import urllib
import requests
import nltk
import torch
from typing import Union
from sentence_transformers import SentenceTransformer, util
from concurrent.futures import ThreadPoolExecutor, as_completed


class GoogleSearch:
    def __init__(self, query: str) -> None:
        self.query = query
        escaped_query = urllib.parse.quote_plus(query)
        self.URL = f"https://www.google.com/search?q={escaped_query}"

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3538.102 Safari/537.36"
        }
        self.links = self.get_initial_links()
        self.all_page_data = self.all_pages()

    def clean_urls(self, anchors: list[str]) -> list[str]:

        links: list[str] = []
        for a in anchors:
            links.append(
                list(filter(lambda l: l.startswith("url=http"), a["href"].split("&")))
            )

        links = [
            link.split("url=")[-1]
            for sublist in links
            for link in sublist
            if len(link) > 0
        ]

        return links

    def read_url_page(self, url: str) -> str:

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(strip=True)

    def get_initial_links(self) -> list[str]:
        """
        scrape google for the query with keyword based search
        """
        print("Searching Google...")
        response = requests.get(self.URL, headers=self.headers)
        soup = BeautifulSoup(response.text, "html.parser")
        anchors = soup.find_all("a", href=True)
        return self.clean_urls(anchors)

    def all_pages(self) -> list[tuple[str, str]]:

        data: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=4) as executor:

            future_to_url = {
                executor.submit(self.read_url_page, url): url for url in self.links
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    output = future.result()
                    data.append((url, output))

                except requests.exceptions.HTTPError as e:
                    print(e)

        # for url in self.links:
        #     try:
        #         data.append((url, self.read_url_page(url)))
        #     except requests.exceptions.HTTPError as e:
        #         print(e)

        return data


class Document:

    def __init__(self, data: list[tuple[str, str]], min_char_len: int) -> None:
        """
        data : list[tuple[str, str]]
            url and page data
        """
        self.data = data
        self.min_char_len = min_char_len

    def make_min_len_chunk(self):
        raise NotImplementedError

    def chunk_page(
        self,
        page_text: str,
    ) -> list[str]:

        min_len_chunks: list[str] = []
        chunk_text = nltk.tokenize.sent_tokenize(page_text)
        sentence: str = ""
        for sent in chunk_text:
            if len(sentence) > self.min_char_len:
                min_len_chunks.append(sentence)
                sent = ""
                sentence = ""
            else:
                sentence += sent
        return min_len_chunks

    def doc(self) -> tuple[list[str], list[str]]:
        print("Creating Document...")
        chunked_data: list[str] = []
        urls: list[str] = []
        for url, dataitem in self.data:
            data = self.chunk_page(dataitem)
            chunked_data.append(data)
            urls.append(url)

        chunked_data = [chunk for sublist in chunked_data for chunk in sublist]
        return chunked_data, url


class SemanticSearch:
    def __init__(
        self, doc_chunks: tuple[list, list], model_path: str, device: str
    ) -> None:

        self.doc_chunks, self.urls = doc_chunks
        self.st = SentenceTransformer(
            model_path,
            device,
        )

    def semantic_search(self, query: str, k: int = 10):
        print("Searching Top k in document...")
        query_embeding = self.get_embeding(query)
        doc_embeding = self.get_embeding(self.doc_chunks)
        scores = util.dot_score(a=query_embeding, b=doc_embeding)[0]

        top_k = torch.topk(scores, k=k)[1].cpu().tolist()
        return [self.doc_chunks[i] for i in top_k], self.urls

    def get_embeding(self, text: Union[list[str], str]):
        en = self.st.encode(text)
        return en
