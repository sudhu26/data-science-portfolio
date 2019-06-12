# this web scraping program compiles data from a baseball reference website

import scrapy
class PlayerInfo(scrapy.Item):
    name = scrapy.Field()
    age = scrapy.Field()
    year = scrapy.Field()
    team = scrapy.Field()
    league = scrapy.Field()
    games = scrapy.Field()
    plateAppearances = scrapy.Field()
    atBats = scrapy.Field()
    runs = scrapy.Field()
    hits = scrapy.Field()
    doubles = scrapy.Field()
    triples = scrapy.Field()
    homeRuns = scrapy.Field()
    runsBattedIn = scrapy.Field()
    stolenBases = scrapy.Field()
    caughtStealing = scrapy.Field()
    baseOnBalls = scrapy.Field()
    strikeouts = scrapy.Field()
    battingAverage = scrapy.Field()
    onBasePercentage = scrapy.Field()
    totalBases = scrapy.Field()
    sluggingPercentage = scrapy.Field()
    doublePlaysGroundedInto = scrapy.Field()
    hitByPitch = scrapy.Field()
    sacrificeHits = scrapy.Field()
    intentionalBaseOnBalls = scrapy.Field()

class MySpider(scrapy.Spider):
    name = 'myspider'
    start_urls = ['https://www.baseball-reference.com/leagues/MLB/2018.shtml']
    
    def parse(self, response):
        for a_el in response.css('div#content table#teams_standard_batting tbody tr'):
            next_page = response.css('div#content table#teams_standard_batting tbody tr a::attr(href)')
            for page in next_page:
                teamPage = response.urljoin(page.extract())
                yield scrapy.Request(teamPage, callback = self.parseTeam)
    
    def parseTeam(self, response):
        next_page = response.css('div#all_team_batting table#team_batting tbody td a::attr(href)')
        for page in next_page:
            playerPage = response.urljoin(page.extract())
            yield scrapy.Request(playerPage, callback = self.parsePlayer)

    def parsePlayer(self, response):
        for playerUrl in response.css('div#wrap'):
            for yr in response.css('table#batting_standard tbody tr:not([class = "minors_table hidden"])'):
                item = PlayerInfo()
                item['name'] = playerUrl.css('div#info div#meta h1::text').extract()
                item['age'] = yr.css('td.right[data-stat = "age"]::text').extract()
                item['year'] = yr.css('th.left[data-stat = "year_ID"]::text').extract()
                item['team'] = yr.css('td.left[data-stat = "team_ID"] a::text').extract()
                item['league'] = yr.css('td.left[data-stat = "lg_ID"] a::text').extract()
                item['games'] = yr.css('td.right[data-stat = "G"]::text').extract()
                item['plateAppearances'] = yr.css('td.right[data-stat = "PA"]::text').extract()
                item['atBats'] = yr.css('td.right[data-stat = "AB"]::text').extract()
                item['runs'] = yr.css('td.right[data-stat = "R"]::text').extract()
                item['hits'] = yr.css('td.right[data-stat = "H"]::text').extract()
                item['doubles'] = yr.css('td.right[data-stat = "2B"]::text').extract()
                item['triples'] = yr.css('td.right[data-stat = "3B"]::text').extract()
                item['homeRuns'] = yr.css('td.right[data-stat = "HR"]::text').extract()
                item['runsBattedIn'] = yr.css('td.right[data-stat = "RBI"]::text').extract()
                item['stolenBases'] = yr.css('td.right[data-stat = "SB"]::text').extract()
                item['caughtStealing'] = yr.css('td.right[data-stat = "CS"]::text').extract()
                item['baseOnBalls'] = yr.css('td.right[data-stat = "BB"]::text').extract()
                item['strikeouts'] = yr.css('td.right[data-stat = "SO"]::text').extract()
                item['battingAverage'] = yr.css('td.right[data-stat = "batting_avg"]::text').extract()
                item['onBasePercentage'] = yr.css('td.right[data-stat = "onbase_perc"]::text').extract()
                item['totalBases'] = yr.css('td.right[data-stat = "TB"]::text').extract()
                item['sluggingPercentage'] = yr.css('td.right[data-stat = "slugging_perc"]::text').extract()
                item['doublePlaysGroundedInto'] = yr.css('td.right[data-stat = "GIDP"]::text').extract()
                item['hitByPitch'] = yr.css('td.right[data-stat = "HBP"]::text').extract()
                item['sacrificeHits'] = yr.css('td.right[data-stat = "SH"]::text').extract()
                item['intentionalBaseOnBalls'] = yr.css('td.right[data-stat = "IBB"]::text').extract()
                yield item   

