mypersonality.csv
 corpus for Personality Recognition from Social Networks:
 includes authors, Facebook statuses in raw text, 
 gold standard labels (both classes and scores) and several
 social network  measures.
 Texts have been originally collected by David Stillwell and
 Michal Kosinski, and anonymized by Fabio Celli.
 Each proper name of person has been replaced with a *PROPNAME*
 string. Famous names, such as "Chopin" and "Mozart", and 
 locations,  such as "New York" and "Mexico", have not been
 replaced.

gold standard labels include:
sEXT	extraversion (score)
sNEU	neuroticism (score)
sAGR	agreableness (score)
sCON	conscientiousness (score)
sOPN	openness (score)

cEXT	extraversion (label: y=extravert, n=shy)
cNEU	neuroticism (label: y=neurotic, n=secure)
cAGR	agreableness (label: y=friendly, n=uncooperative)
cCON	conscientiousness (label: y=precise, n=careless)
cOPN	openness (label: y=insightful, n=unimaginative)