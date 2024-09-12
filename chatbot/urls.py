from django.urls import path
from . import views

app_name = "chatbot"

urlpatterns = [
    path('', views.login, name="index"),
    path('chatroom/', views.chatroom, name="chatroom"),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('chatroom/<int:pk>/', views.prev_chat, name='prev_chat'),
    path('chatroom/save/', views.chat_save, name='chat_save'),
    path('chatroom/msg_save/', views.message_save, name='msg_save'),
]