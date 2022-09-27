from django.shortcuts import render

# Create your views here.
from .models import HouseFeatures

# Create your views here.
def home_view(request):
    print(request.method)
    context={}
    if request.method == "POST":
        params=request.POST
        newHouse=HouseFeatures(**params)
        res=newHouse.predict_val()
        context={
            'Lowr':res[1],
            'Highr':res[0]
        }
    return render(request,'home_view.html',context)

def pred_view(request):
    pass
