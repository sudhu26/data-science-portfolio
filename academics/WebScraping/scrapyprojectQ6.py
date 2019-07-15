import scrapy

class DocSectionItem(scrapy.Item):
    title = scrapy.Field()
    company = scrapy.Field()
    location = scrapy.Field()


class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://stackoverflow.com/jobs?med=site-ui&ref=jobs-tab']
    
    def parse(self, response):
        for a_el in response.css('div#mainbar div.listResults div.-job-summary'):
            section = DocSectionItem()
            section['title'] = a_el.css('div.-title a::text').extract()[0]
            section['company'] = a_el.css('span:not(.fc-black-500)::text').extract()[2].strip().replace('\r\n','')
            section['location'] = a_el.css('span.fc-black-500::text').extract()[1].strip().replace('- \r\n','')
            yield section
            
            next_page = response.css('div#mainbar div.pagination a.prev-next.job-link.test-pagination-next::attr(href)')[0]
            if next_page is not None:
                next_page = response.urljoin(next_page.extract())
                yield scrapy.Request(next_page, callback = self.parse)

