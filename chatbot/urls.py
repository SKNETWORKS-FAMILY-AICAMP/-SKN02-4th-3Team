from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "chatbot"

urlpatterns = [
    path('', views.login, name="index"),
    path('chatroom/', views.chatroom, name="chatroom"),
    path('login/', auth_views.LoginView.as_view(template_name='chatbot/login.html'), name='login'),
    path('logout/', views.logout, name='logout'),
    path('chatroom/<int:chat_id>/', views.prev_chat, name='prev_chat'),
    path('chatroom/save/', views.chat_save, name='chat_save'),
    path('chatroom/msg_save/', views.message_save, name='msg_save'),
    path('chatroom/chatbot_answer/', views.chatbot_response, name="chatbot_answer")
]