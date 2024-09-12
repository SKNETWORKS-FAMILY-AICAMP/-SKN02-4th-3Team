from django.http import HttpResponse, JsonResponse
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.forms.models import model_to_dict
from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render, redirect
from .openai_chatbot import get_answer_from_rag_chain as chatbot

import json

import markdown
from django.utils.safestring import mark_safe

from .models import Chat, Message

def login(request):
    if request.user.is_authenticated:
        return redirect('/chatroom')
    else:
        return render(request, "chatbot/login.html")

def logout(request):
    auth_logout(request)
    return redirect('/login')

@login_required(login_url='/login')
def chatroom(request):
    chats = Chat.objects.filter(user=request.user.username, saved_yn=True).values()
    chat_list = []
    for chat in chats:
        m_q = Message.objects.filter(chat_id=chat['id'], sender="user").order_by('pk').values()[0]
        m_a = Message.objects.filter(chat_id=chat['id'], sender="chatbot").order_by('pk').values()[0]
        chat_list.append({"chat": chat, "question": m_q, "answer": m_a})

    new_chat = Chat.objects.create(user=request.user.username)
    new_chat.save()
    chat_id = new_chat.id
    # return render(request, "chatbot/chatroom.html")
    return render(request, "chatbot/chatroom.html", {"chat_id": chat_id, "chat_list": chat_list})

@login_required(login_url='/login')
def prev_chat(request, chat_id):
    username = request.user.username
    
    chat = Chat.objects.get(id=chat_id)

    if username != chat.user:
        return redirect('/chatroom')

    message_list = Message.objects.filter(chat_id=chat_id).order_by('pk').values()
    chat_list = get_chat_list(username)

    return render(request, "chatbot/chatroom.html", {"chat_id": chat_id, "message_list": message_list, "chat_list": chat_list})

@login_required(login_url='/login')
def chat_save(request):
    body = json.loads(request.body)
    chat_id = body["chat_id"]
    c = Chat.objects.get(pk=chat_id)
    c.saved_yn = True
    c.save()
    return redirect('/chatroom')

@login_required(login_url='/login')
def message_save(request):
    body = json.loads(request.body)
    m = Message.objects.create(chat_id=body["chat_id"], sender=body["sender"], content=body["content"], markdown=body["markdown"])
    m.save()
    data = model_to_dict(m)
    time_string = json.dumps(m.sended_at.isoformat())
    data['time_str'] = time_string
    return JsonResponse(data)

def chatbot_response(request):
    body = json.loads(request.body)
    user_msg = body["user_msg"]

    response = chatbot(user_msg)

    extensions = ["nl2br", "fenced_code"]
    res_md = mark_safe(markdown.markdown(response, extensions=extensions))

    data = {"response": response, "markdown": res_md}
    return JsonResponse(data)

def get_chat_list(username):
    chats = Chat.objects.filter(user=username, saved_yn=True).values()
    chat_list = []
    for chat in chats:
        m_q = Message.objects.filter(chat_id=chat['id'], sender="user").order_by('pk').values()[0]
        m_a = Message.objects.filter(chat_id=chat['id'], sender="chatbot").order_by('pk').values()[0]
        chat_list.append({"chat": chat, "question": m_q, "answer": m_a})
    return chat_list