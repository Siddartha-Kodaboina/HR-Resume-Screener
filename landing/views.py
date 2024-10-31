from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages

# Create your views here.
def landing_page(request):
    return render(request, 'landing/landing.html')

def recruiter_login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        #user = authenticate(request, username=email, password=password)

        # if user is not None:
            # login(request, user)
            # messages.success(request, "Login successful!")
            # Redirect to Chainlit after successful login
        return redirect('ml_engineer_job')  
        # else:
        #     messages.error(request, "Invalid email or password.")
    
    return render(request, 'login/login.html')

# View to render the job description page
def ml_engineer_job_view(request):
    return render(request, 'ml-engineer-job/ml-engineer-job.html')

# View to redirect to Chainlit running on port 9000
def chainlit_redirect(request):
    # Redirect to Chainlit service with the email as a query parameter
    chainlit_url = f'http://127.0.0.1:9000'
    return redirect(chainlit_url)
