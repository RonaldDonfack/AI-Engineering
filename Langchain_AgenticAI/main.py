from itertools import chain
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():
    print("Hello Langchain AgenticAI")
    information = """Carlos Alcaraz Garfia[a] (born 5 May 2003) is a Spanish professional tennis player. He was ranked world No. 1 in men's singles by the Association of Tennis Professionals (ATP), and finished as the year-end No. 1 in 2022 and 2025. Alcaraz has won 26 ATP Tour–level singles titles, including seven majors and eight ATP Masters 1000 titles. He is the youngest man to complete the career Grand Slam in singles.

Alcaraz began his tennis career in 2018 at age 14. He broke into the top 100 of the rankings in May 2021, and ended that year ranked No. 32 after reaching the US Open quarterfinals. In 2022, Alcaraz won his first major title at the US Open. He became the youngest man and the first male teenager in the Open Era to be ranked world No. 1, at 19 years, 4 months and 7 days old, and finished the year as the youngest year-end No. 1 in ATP rankings history.

Alcaraz claimed his second major title at Wimbledon in 2023, defeating seven-time champion Novak Djokovic in the final. In 2024, he became the youngest man in the Open Era to complete the Channel Slam (winning both the French Open and Wimbledon titles in one season), and won a silver medal at the Paris Olympics. Alcaraz claimed his fifth and sixth major titles in 2025, defeating Jannik Sinner in the final of the French Open as well as the US Open. In 2026, Alcaraz won the Australian Open, becoming the youngest man in history to complete the Career Grand Slam, at 22 years, 8 months and 27 days old.

Early and personal life
Carlos Alcaraz Garfia was born on 5 May 2003, in El Palmar, Murcia, Spain, to parents Carlos Alcaraz González and Virginia Garfia Escandón. Alcaraz has one older brother, Álvaro, and two younger brothers, Sergio and Jaime.[6]

Alcaraz started playing tennis at the age of four at the Real Sociedad Club de Campo de Murcia, where his father was a tennis coach and club administrator.[7] His mother worked as a sales assistant at IKEA.[8] Alcaraz's father had played tennis but stopped pursuing a professional career as a teenager, as he could not afford it.[9][10] In 2013, at age 10, Alcaraz signed his first contract with Babolat.[11] After watching him compete at age 11, IMG agent Albert Molina recognized Carlos as a standout talent and sought to persuade his parents to work with him; his father ultimately agreed when Carlos was 12.[12][13][14]

Alcaraz is frequently accompanied to tournaments by his father and by his brother Álvaro, who works as his hitting partner and assistant coach. During the tennis off-season, he lives at his parents' home in Murcia.[15] His friends and family call him "Carlitos" or "Charly".[16] He is Catholic, and has received blessings from priests before important tournaments.[17][18]
 At the age of 10, Alcaraz competed outside Spain for the first time at the under-10 World Championship in Croatia,[19] where he lost in the final.[20] That same year, he became an official member of the Babolat team, later joining the brand's international roster at age 13.[21]

In 2015, Alcaraz won the under-12 division of the Rafa Nadal Tour Masters, and the following year captured the competition's under-14 title.[22]

In 2017, Alcaraz enjoyed a breakthrough season at the under-14 level. He won the XIV Taça Internacional Maia Jovem, defeating Daniel Rincón in the final,[23] and later captured the Babolat Cup.[24] That summer, he helped Spain win the 14-and-under European Summer Cup[25] and was part of the Spanish team that finished runner-up to Switzerland at the ITF World Junior Tennis Finals.[26]

Alcaraz continued his rise in 2018. In July, he won the Dutch Junior Open, defeating Filip Kolasinski in the final,[27] and weeks later captured the European 16-and-under Championship by defeating Elmer Møller.[28] Later that year at the Junior Davis Cup in Budapest, he played a key role in Spain's title run, saving match point while trailing against Harold Mayot in singles before returning to win the decisive doubles match after France had taken an early lead.[29][30]

In March 2019, Alcaraz won the J300 Villena, defeating Illya Beloborodko in the final,[31][32] and was ranked No. 1 on the Tennis Europe Junior Tour during the year.[33] As a junior, he achieved a career-high world ranking of No. 22.[34] """


    summary_template =  """Given the information{information} about a person I want you to create : 
    1. A short summary
    2. Two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gpt-oss:20b")
    # llm = ChatOllama(temperature=0, model="gemma4:e4b")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()