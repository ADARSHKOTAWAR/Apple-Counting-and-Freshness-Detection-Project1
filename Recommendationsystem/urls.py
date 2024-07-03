"""Recommendationsystem URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import index
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls), 
    path('page1',index.page1),
    path('aboutus',index.aboutus, name='aboutus'),
    path('register',index.register),
    path('ourteam',index.ourteam),
    path('contact',index.contact, name='contact'),
    path('',index.login),
    path('doregister',index.doregister),
    path('calculate',index.calculate),
    path('recommend',index.recommend),
    path('dologin',index.dologin),
    path('viewuser',index.viewuser),
    path('logout',index.logout, name='logout'),
    path('prevpred',index.prevpred),
    path('temp',index.temp),
    path('index',index.index),
    path('analyze',index.analyze),
    path('userhome',index.userhome, name = 'userhome'),
    path('adminhome',index.adminhome),
    path('count',index.count, name='count'),
    path('quality',index.quality, name="quality"),
    path('detect_apples',index.detect_apples, name="detect_apples"),
    path('process_image',index.process_image, name="process_image"),
    path('process_session_image', index.process_session_image, name='process_session_image'),
    path('viewpredicadmin',index.viewpredicadmin),
    path('detect_apples_with_yolo',index.detect_apples_with_yolo,name='detect_apples_with_yolo'),
    path('vgg_quality',index.vgg_quality,name='vgg_quality'),
    path('resnet_quality',index.resnet_quality,name='resnet_quality'),
    path('cnn_quality',index.cnn_quality,name='cnn_quality'),
    path('show_accuracies_graph/', index.show_accuracies_graph, name='show_accuracies_graph'),
    path('proposed_quality', index.proposed_quality, name='proposed_quality'),
    # path('detect_and_predict_apples_vgg16/', index.detect_and_predict_apples_vgg16, name='detect_and_predict_apples_vgg16'),
    # path('detect_and_predict_apples_resnet50/', index.detect_and_predict_apples_resnet50, name='detect_and_predict_apples_resnet50'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

