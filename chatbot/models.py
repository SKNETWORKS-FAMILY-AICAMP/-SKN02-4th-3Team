from django.db import models

class Chat(models.Model):
    # User ID
    user = models.CharField(max_length=20)

    # Create_at
    created_at = models.DateTimeField(auto_now_add=True)

    # Saved Chatting
    saved_yn = models.BooleanField(default=False)

class Message(models.Model):
    # Chat Object -- access by chat_id
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    
    # 'chatbot' or 'user'
    sender = models.CharField(max_length=10)

    # Chatting content
    content = models.TextField()

    # Message Sended Time
    sended_at = models.DateTimeField(auto_now_add=True)