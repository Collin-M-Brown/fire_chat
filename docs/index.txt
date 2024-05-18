# AI: Man's new best friend

![alt text](download.png)

# Table of Contents: 
- AI Friend
- Basic Components
    - STT
    - TTT
    - TTS
- Addons

# AI Friend
In this guide I will be going over how to have a conversation with your AI. When I say conversation, I don't mean simply handling audio and text conversions; I mean having a realistic, human-like, fast-paced back-and-forth dialogue. 

The most basic "conversation" with an AI consists of a Q and A session. User speaks, AI responds, User speaks, AI responds... etc...

But real conversations don't follow a linear pattern. Sometimes human's respond instantly; sometimes they take a while to think. They can interrupt your or zone out when you're speaking. You might bring up a topic A but they might be fixated on topic B. They can agree with you when you are wrong, and disagree when you are right.

An AI that always listens to you and returns consistant, factual, and respectful responses will be a useful addition to your toolbox; but could you really call it a friend?

# Basic Components
First I will go over the basic building blocks for your AI friend.
1. Speech-to-text (STT)
2. Text-to-text (TTT aka your large language model)
3. Text-to-speech (TTS)

# Setup 
For this guide I you will need three API keys, but all three companies offer free starting credits.
1. Deepgram API (STT) https://deepgram.com
2. Fireworks API (TTT) https://fireworks.ai
3. OpenAi API (TTS) https://openai.com
I will also be going over some alternatives at the end of this guide.

### 1. Place your API keys in a .env file in your working directory
Here are my keys as an example. Please DO NOT use them.
```
#.env
export OPENAI_API_KEY='sk-4Qy7t8Lz1Mn3Wk5R2Xv9Hs6P0BqTj8GcD4F1Ra2'
export FIREWORKS_API_KEY='uB9x8Gz2Lw5Np7Jm3Tc6Vd4P1Qk7R1Xo2Ys8Hn5M'
export DEEPGRAM_API_KEY='G6p9Bz2Lw1Qk7R8Xv3Tj5Hs0P4R9Wn3Yc2F8D1Ka'
```

### 2. Create a fresh environment
```bash

```

# Speech-to-text
There are a number of tools and API's that you can use to convert your speech to text but for this demonstration I will be using Deepgram API with their Nova model. They give you $200 starting credits for free. For personal use, this will last you a very... very long time.

# Bringing your new friend to life
