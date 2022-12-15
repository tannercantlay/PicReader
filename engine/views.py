from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .forms import UploadFileForm

#from somewhere import my image parser

def index(request):
	if request.method == 'POST':
		form = UploadFileForm(request.Post, request.FILES)
		if form.is_valid():
			handle_uploaded_file(request.FILES['files'])
			return HttpResponseRedirect('/success/url/')
	else:
		form = UploadFileForm()
	return render(request, 'templates/index.html', {'form': form})
