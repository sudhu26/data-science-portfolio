import scrapy

class DocTitleItem(scrapy.Item):
    page_title = scrapy.Field()
    page_url = scrapy.Field()
    
class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://stackoverflow.com/jobs?med=site-ui&ref=jobs-tab']
    
    def parse(self, response):
        for follow_href in response.css('div#mainbar div.listResults div.-job-summary div.-title a::attr(href)'):
            follow_url = response.urljoin(follow_href.extract())
            yield scrapy.Request(follow_url, callback = self.parse_page_title)

    def parse_page_title(self, response):
        doc = DocTitleItem()
        doc['page_title'] = response.css()
        doc['page_url'] = response.url
        yield doc


section['url_link'] = a_el.css('div.-title a::attr(href)').extract()[0]
            