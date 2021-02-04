from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Python program to generate WordCloud

# Reads 'Youtube04-Eminem.csv' file  
text = 'Hi, my name is mo. worlds cloud'

positive = 'lifesaver goooood pleasantly fantabulous painless yummmy delicious stupendous invaluable goood addicting excellent reasonable masterful phenom cutest magnificent wac prefect heavenly'
negative = 'unacceptable poorest withdrew blandest disgusting unprofessionally inexcusable unacceptably untruthful discriminates disgrace ridicules ruder slooooow scowled uninspiring disrespected deplorable mismanaged unreliable'

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=20).generate(negative)

# plot the WordCloud image                        
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
