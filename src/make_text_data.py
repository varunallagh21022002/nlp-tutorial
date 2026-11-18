"""Original teaching corpora, authored for this exercise (not real user data)."""
import csv
from pathlib import Path
DATA = Path(__file__).resolve().parents[1] / 'data'
CORPORA = {
'sms': {
'ham': '''Meet me at the library after class
Please bring your notebook tomorrow
Dinner is ready come home soon
Can you send the lecture notes
The meeting has moved to Friday
Your dentist appointment is confirmed
I will call you after work
Please pick up milk on your way
Happy birthday have a lovely day
Our train arrives this evening
The project deadline is next week
Are you coming to football practice
I left your keys on the desk
Thanks for helping with my homework
Let us have lunch together today
The office will be closed tomorrow
Your package will arrive on Monday
Remember to take your umbrella
The teacher uploaded the assignment
Can we reschedule our study session
My phone battery is almost empty
Please check the document I sent
We are waiting near the station
The family picnic starts at noon
I enjoyed our conversation yesterday
Call the plumber about the kitchen sink
Your prescription is ready for collection
See you at the conference next month
Please water the plants while I travel
The bus is running late today
Bring your sports shoes for practice
Your table reservation is confirmed
The laptop charger is in my bag
Please review the budget before our meeting
The exam room has been changed
I booked tickets for our weekend trip
We should discuss the report after lunch
The grocery list is on the fridge
Send me the address for your house
Your friend is waiting at reception''',
'spam': '''Congratulations claim your free cash prize now
You won a lottery click here to collect
Urgent cash reward waiting call immediately
Exclusive offer win a luxury holiday today
Get a free phone by entering this contest
Claim your guaranteed jackpot before midnight
Earn money fast with our secret system
Your number was selected for a cash bonus
Limited offer buy now and get free gifts
Click to unlock an exclusive reward voucher
Winner claim your prize by paying a small fee
Free vacation awaits respond with your details
Double your money with this amazing investment
Congratulations you are our lucky winner today
Instant loan approved claim cash without documents
Buy discount pills from our online store
You have won a brand new sports car
Collect your reward by calling this premium line
Get rich quickly with no work required
Special bonus for you claim before it expires
Win free shopping vouchers sign up now
Urgent prize notification send your bank details
Exclusive casino bonus claim your free spins
Cash giveaway selected winners respond immediately
Amazing discount deal only for lucky customers
Claim a free gift card at this link
Guaranteed investment returns double your savings
Your account qualifies for a surprise cash reward
Free holiday tickets claim with a processing fee
Congratulations collect your unlimited shopping credit
Cheap luxury watches available buy today
Win a fortune with our lottery promotion
Redeem your exclusive prize voucher immediately
Get instant cash by joining our reward scheme
Lucky draw winner send details to claim
Your free subscription prize is ready now
Make money from home guaranteed daily profits
Final chance to claim your luxury gift
Unlock a secret jackpot by clicking our link
Cash prize expires tonight reply winner now'''
},
'reviews': {
'positive': '''A wonderful story with excellent acting
The film was charming and beautifully directed
I loved the clever dialogue and warm characters
An exciting adventure with a satisfying ending
The performances were superb throughout the movie
A moving story that stayed with me
Brilliant cinematography made every scene memorable
This comedy was funny and full of heart
The soundtrack was beautiful and the acting convincing
A delightful film worth watching again
An intelligent script supported by excellent actors
The pacing was energetic and the plot engaging
I enjoyed the emotional depth of the characters
A powerful performance lifted an already great story
The movie was surprisingly original and entertaining
A lovely family film with a thoughtful message
The director created a vivid and wonderful world
Excellent action scenes and a compelling story
The ending was touching and deeply satisfying
A clever mystery with believable characters
The actors delivered warm and memorable performances
A stunning visual experience with real emotional weight
I was impressed by the sharp writing
The film made me laugh and care about everyone
A fantastic adaptation with beautiful music
The story was absorbing from the opening scene
A sensitive and honest portrayal of friendship
The comedy timing was excellent and natural
This was an enjoyable and rewarding cinema experience
A gripping thriller with brilliant performances
The animation was gorgeous and imaginative
A great cast brought the story to life
The film was not boring and kept me engaged
A hopeful ending made this a pleasant watch
The lead actor was superb in a difficult role
A refreshing story with charming humour
The dialogue felt authentic and the direction confident
A masterful blend of suspense and emotion
I recommend this wonderful and uplifting movie
A thoughtful drama with outstanding character development''',
'negative': '''A boring story with terrible acting
The film was dull and poorly directed
I hated the awkward dialogue and flat characters
A tedious adventure with a disappointing ending
The performances were weak throughout the movie
A confusing story that never made sense
Poor editing made every scene feel endless
This comedy was unfunny and painfully predictable
The soundtrack was annoying and the acting wooden
A dreadful film not worth watching again
A lazy script wasted a talented cast
The pacing was slow and the plot uninteresting
I disliked the shallow portrayal of the characters
A weak performance ruined an already poor story
The movie was painfully repetitive and forgettable
A noisy family film with an empty message
The director created a lifeless and confusing world
Terrible action scenes and an incoherent story
The ending was abrupt and deeply unsatisfying
A predictable mystery with unbelievable characters
The actors delivered stiff and forgettable performances
An ugly visual experience with no emotional impact
I was disappointed by the clumsy writing
The film failed to make me laugh or care
A bad adaptation with irritating music
The story was tedious from the opening scene
A shallow and dishonest portrayal of friendship
The comedy timing was awkward and forced
This was a frustrating and unrewarding cinema experience
A dull thriller with terrible performances
The animation was crude and uninspired
A weak cast failed to bring the story to life
The film was not good and left me disappointed
A rushed ending made this an unpleasant watch
The lead actor was unconvincing in every scene
A stale story with childish humour
The dialogue felt artificial and the direction careless
A messy blend of cheap suspense and melodrama
I cannot recommend this boring and depressing movie
A pointless drama with poor character development'''
},
'news': {
'Sports': '''Football team wins championship after dramatic final
Cricket captain scores century in tournament victory
Tennis star reaches final after thrilling match
Olympic runner sets record in sprint race
Basketball team signs experienced coach for season
Hockey players prepare for international tournament
Football club announces new stadium training facilities
Swimmer wins gold medal at national championships
Cricket league introduces revised match schedule
Tennis champion returns after injury recovery
Athletics federation selects squad for world games
Baseball team celebrates playoff victory at home
Cyclist takes lead in mountain stage race
Badminton player wins first international title
Rugby squad prepares for difficult away match
Football goalkeeper saves penalty in cup final
Marathon runners gather for annual city race
Volleyball team secures place in regional finals
Boxing champion defends title in close contest
Sports academy launches training camp for young athletes''',
'Business': '''Stock market rises as company profits beat forecasts
Bank announces lower interest rates for business loans
Retail company reports strong quarterly sales growth
Investors welcome merger between major financial firms
Oil prices fall amid concerns about global demand
Startup raises funding to expand its operations
Central bank reviews inflation and monetary policy
Manufacturing company opens factory creating new jobs
Currency markets react to economic growth figures
Export businesses seek relief from rising costs
Shareholders approve new corporate investment plan
Banking sector reports increase in digital payments
Retail chain expands stores ahead of festive season
Insurance company announces improved annual earnings
Investors shift funds as bond yields increase
Small businesses report recovery in customer demand
Trade agreement aims to boost exports and investment
Company shares decline after disappointing revenue report
Economic survey shows growth in service industries
Finance ministry discusses tax incentives for businesses''',
'Technology': '''Software company launches new artificial intelligence tool
Researchers develop faster computer chip for mobile devices
Cybersecurity experts discover flaw in popular software
New smartphone offers improved battery and camera
Cloud computing platform adds data protection features
Scientists test robot designed for underwater exploration
Technology startup builds accessible virtual reality headset
New programming framework simplifies web application development
Artificial intelligence system improves speech recognition accuracy
Space agency tests communication technology for satellites
Computer engineers develop energy efficient processors
Software update fixes security issues in messaging app
Researchers demonstrate advances in quantum computing hardware
New wearable device monitors fitness with smart sensors
Internet provider expands high speed fibre network
Robotics team unveils automated warehouse machine
Technology firm releases open source machine learning library
Smartphone manufacturer introduces foldable display design
Cybersecurity conference focuses on protecting digital identities
Developers create software to detect manipulated images''',
'Politics': '''Parliament debates new education bill during winter session
Government announces reforms after cabinet meeting
Election commission publishes schedule for national polls
Opposition leaders demand debate on public spending
President meets foreign minister for diplomatic talks
Voters head to polls in closely contested election
Political parties negotiate coalition after election results
Cabinet approves proposal for local government reform
Lawmakers question minister during parliamentary session
Government introduces legislation on public administration
Prime minister addresses nation on policy priorities
Election campaign focuses on jobs and public services
Parliamentary committee reviews proposed constitutional amendment
Diplomatic delegation discusses regional peace agreement
Opposition party announces candidates for upcoming election
State government presents new transport policy in assembly
Lawmakers vote on reforms after lengthy debate
Political leaders attend summit on international cooperation
Election officials announce updated voter registration rules
Cabinet minister resigns following parliamentary inquiry''',
'Entertainment': '''Film festival opens with premiere of acclaimed drama
Actor announces new movie with award winning director
Singer releases album ahead of international concert tour
Television series returns with new season and cast
Music awards celebrate outstanding performances of the year
Popular actress stars in upcoming romantic comedy
Theatre company stages classic play for summer season
Streaming platform announces slate of original films
Fans welcome musician at sold out concert venue
Animated film tops weekend box office chart
Director discusses creative process at cinema festival
Celebrity couple attends charity gala and film premiere
New documentary explores history of popular music
Comedy show brings famous performers to city theatre
Actor wins award for leading role in historical film
Dance festival features performances from international artists
Movie studio releases trailer for fantasy adventure
Singer announces acoustic concerts across major cities
Television drama receives praise for ensemble cast
Entertainment awards honour veteran filmmaker and musicians'''
}}

def write_data():
    DATA.mkdir(exist_ok=True)
    for name, classes in CORPORA.items():
        with (DATA / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['text', 'label'])
            for label, block in classes.items():
                writer.writerows((text, label) for text in block.splitlines())
if __name__ == '__main__': write_data()
