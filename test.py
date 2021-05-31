import json
import pandas as pd

# df = pd.read_csv('meditation_playlist.csv')
# df['SpotifyID'] = df['SpotifyID'].map(lambda x: 'http://open.spotify.com/track/' + x.strip())
# df.rename(columns={"SpotifyID":"URL"}, inplace=True)

# df.to_csv('links.csv')

# print(df)

emotions = ['sad', 'bad', 'cat']

emotion = ' and '.join(emotions)
print(emotion)
