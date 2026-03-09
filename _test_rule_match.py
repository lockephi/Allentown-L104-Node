#!/usr/bin/env python3
"""Test rule pattern matching on failing questions."""
import re

rules_to_test = [
    # (rule_q_pattern, question_text, description)
    (r'(?:daylight.+different|hours.+daylight.+differ|daylight.+more|daylight.+less|daylight.+(?:january|february|march))',
     'On January 15th, there were 10 hours and 24 minutes of daylight in Jacksonville, Florida. On the same day, there were only 9 hours and 37 minutes of daylight',
     'daylight/tilt'),
    (r'(?:flagell|cilia|volvox|paramecium).+(?:help|function|purpose)',
     'A euglena has a flagellum and a paramecium has cilia. Both types of organelles help these organisms move through the water. However, a paramecium can also use cilia to',
     'flagellum/cilia food'),
    (r'(?:frog|toad).+(?:compet|fight|struggle)',
     'A group of frogs live together near a small pond. The frogs compete with each other most for which resource?',
     'frogs compete'),
    (r'(?:rising.+pacific.+temp|pacific.+drought|cause.+rising.+surface.+temp.+pacific)',
     'What is the cause of rising surface temperatures of the Pacific Ocean, drought in the western United States, and flooding on the Pacific coast of South America?',
     'El Nino cause'),
    (r'(?:conduction\s+occurs|heat.+conduction|conduction.+when\s+molecule)',
     'Heat transfer by conduction occurs when molecules',
     'conduction'),
    (r'(?:30%.+less\s+fat|less\s+fat|reduced\s+fat).+(?:infer|claim)',
     'What can be inferred from a food product advertisement that claims "30% less fat than our leading competitors"?',
     '30% less fat'),
    (r'(?:perspiration|sweat|sweating).+(?:role|purpose|function|primary)',
     'Which statement best describes the primary role of perspiration in humans?',
     'perspiration'),
    (r'(?:neutraliz|acid.+base.+react|HCl.+NaHCO|double\s+replacement)',
     'Sodium bicarbonate (NaHCO_{3}) will neutralize stomach acid (HCl) in a double replacement reaction',
     'neutralization H2O'),
    # Missing: kick contact force
    (r'(?:kick.+ball|ball.+kick|ball.+move.+ground|why.+ball\s+move)',
     'A student places a ball on the ground and kicks it. The ball moves along the ground. Why does the ball move?',
     'kick/contact force'),
    # Missing: industrial gases
    (r'industrial\s+gas.+atmosphere|gas.+released.+atmosphere',
     'Large amounts of industrial gases are released into the atmosphere every day. What happens to those gases?',
     'industrial gases'),
    # Missing: meteorologists study weather
    (r'meteorolog.+(?:study|know|should|learn)',
     'Meteorologists study weather. Which of the following should meteorologists know about?',
     'meteorologists/fronts'),
    # Missing: birds identification
    (r'(?:birds?.+park|(?:many|different)\s+kinds?.+birds?)',
     'A student wants to find out how many different kinds of birds are found in a park. In addition to a bird identification book, the student should use _',
     'bird identification'),
    # Missing: Erosion = move rocks
    (r'(?:erosion.+weathering|weathering.+erosion).+(?:happen|occurs?).+(?:only|because)',
     'Erosion and weathering can both cause changes to the surface of Earth. Which of the following happens only because of erosion and NOT because of weathering?',
     'erosion only'),
    # Test: fact vs opinion earthquakes
    (r'(?:fact.+(?:opinion|rather)|opinion.+fact)',
     'Which is a fact rather than an opinion about earthquakes?',
     'fact vs opinion earthquakes'),
    # Test: coach stopwatch
    (r'(?:stopwatch|stop\s+watch|timer.+measur|holding.+stopwatch)',
     'A coach is standing at the finish line of a race. He is holding a stopwatch. What is the coach most likely measuring with the stopwatch?',
     'stopwatch'),
    # Test: skunk/smell
    (r'(?:skunk.+sense|sense.+skunk)',
     'Which of these senses lets Dora know when a skunk has been close to her house?',
     'skunk smell'),
]

for pat, q, desc in rules_to_test:
    m = re.search(pat, q, re.IGNORECASE)
    print(f"{'MATCH' if m else 'MISS '}: [{desc}] pattern: {pat[:70]}...")
