import newspaper

def extract(url):
    config = newspaper.Config()
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    config.request_timeout = 10

    paper = newspaper.build(url, config=config, memoize_articles=False)
    
    for article in paper.articles:
        print(article.url)
    print(len(paper.articles))

extract('https://www.jugantor.com/')
