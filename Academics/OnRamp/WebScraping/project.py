import scrapy

class Teams(scrapy.Item):
    teamName = scrapy.Field()
    teamUrl = scrapy.Field()

class TeamInfo(scrapy.Item):
    playerName = scrapy.Field()

class PlayerInfo(scrapy.Item):
    name = scrapy.Field()
    age = scrapy.Field()
    year = scrapy.Field()
    team = scrapy.Field()
    #a = scrapy.Field()
    

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.baseball-reference.com/leagues/MLB/2018.shtml']
    
    def parse(self, response):
        for a_el in response.css('div#content table#teams_standard_batting tbody tr'):
            #item = Teams()
            #item['teamName'] = a_el.css('a::text').extract()
            #yield item
            
            next_page = response.css('div#content table#teams_standard_batting tbody tr a::attr(href)')
            for page in next_page:
                teamPage = response.urljoin(page.extract())
                yield scrapy.Request(teamPage, callback = self.parseTeam)
    
    def parseTeam(self, response):
        #item = TeamInfo()
        #item['playerName'] = response.css('div#all_team_batting table#team_batting tbody td a::text').extract()
        #yield item

        next_page = response.css('div#all_team_batting table#team_batting tbody td a::attr(href)')
        for page in next_page:
            playerPage = response.urljoin(page.extract())
            yield scrapy.Request(playerPage, callback = self.parsePlayer)

    def parsePlayer(self, response):
        for playerUrl in response.css('div#wrap'):
            item = PlayerInfo()
            
            item['name'] = playerUrl.css('div#info div#meta h1::text').extract()
            item['age'] = playerUrl.css('table#batting_standard tbody tr:not([class = "minors_table hidden"]) td.right[data-stat = "age"]::text').extract()
            item['year'] = playerUrl.css('table#batting_standard tbody tr:not([class = "minors_table hidden"]) th.left[data-stat = "year_ID"]::text').extract()
            item['team'] = playerUrl.css('table#batting_standard tbody tr:not([class = "minors_table hidden"]) td.left[data-stat = "team_ID"]::text').extract()
            yield item

        



