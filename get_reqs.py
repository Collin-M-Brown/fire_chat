with open(r'/home/navi/Repos/fire_chat/requirements.txt', 'r') as req:
    content = req.read()
    requirements = content.split('\n')
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]


print(requirements)