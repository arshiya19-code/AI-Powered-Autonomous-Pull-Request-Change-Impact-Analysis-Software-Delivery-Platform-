"""Opaque, revocable HttpOnly sessions; credentials never leave this module."""
import hashlib
import os
import re
import secrets
import time
import threading
from collections import defaultdict, deque
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from .database import DB, User, Session, PasswordReset, Analysis, AuditEvent, public_user

router = APIRouter(prefix='/api')
hasher = PasswordHasher()
dummy_hash = hasher.hash(secrets.token_urlsafe(24))
COOKIE = 'impact_session'
secure = os.getenv('COOKIE_SECURE', 'false').lower() == 'true'
origins = set(os.getenv('APP_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(','))
attempts = defaultdict(deque)
lock = threading.Lock()
def digest(value): return hashlib.sha256(value.encode()).hexdigest()

def limited(request, scope):
    key = (request.client.host if request.client else 'unknown', scope)
    with lock:
        now = time.time()
        bucket = attempts[key]
        while bucket and bucket[0] < now - 600: bucket.popleft()
        if len(bucket) >= 15: raise HTTPException(429, 'Too many attempts. Try again in ten minutes.')
        bucket.append(now)

def current_user(request: Request):
    token = request.cookies.get(COOKIE)
    with DB() as db:
        session = db.get(Session, digest(token)) if token else None
        user = db.get(User, session.user_id) if session and session.expires_at > time.time() else None
        if not user or not user.is_active: raise HTTPException(401, 'Please sign in')
        return user

def admin(user=Depends(current_user)):
    if user.role != 'ADMIN': raise HTTPException(403, 'Administrator access required')
    return user

def issue(db, user, response):
    token = secrets.token_urlsafe(48)
    db.add(Session(token_hash=digest(token), user_id=user.id, expires_at=time.time()+86400*7))
    response.set_cookie(COOKIE, token, httponly=True, secure=secure, samesite='lax', max_age=86400*7, path='/')

class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=128)
    @field_validator('email')
    @classmethod
    def email_format(cls, v):
        v = v.strip().lower()
        if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', v): raise ValueError('Enter a valid email address')
        return v

def strong(password):
    if len(password) < 12: raise HTTPException(422, 'Use at least 12 characters')

class Registration(Credentials):
    display_name: str = Field(min_length=1, max_length=100)

@router.post('/auth/register', status_code=201)
def register(body: Registration, request: Request, response: Response):
    limited(request, 'register'); strong(body.password)
    with DB() as db:
        user = User(email=body.email, display_name=body.display_name.strip(), password_hash=hasher.hash(body.password))
        if not user.display_name: raise HTTPException(422, 'Display name is required')
        db.add(user)
        try: db.flush()
        except IntegrityError: raise HTTPException(409, 'Unable to create account with this email')
        issue(db, user, response)
        db.add(AuditEvent(actor_id=user.id, action='REGISTER', subject_id=user.id))
        db.commit()
        return public_user(user)

@router.post('/auth/login')
def login(body: Credentials, request: Request, response: Response):
    limited(request, 'login')
    with DB() as db:
        user = db.scalar(select(User).where(User.email == body.email))
        try: valid = hasher.verify(user.password_hash if user else dummy_hash, body.password)
        except VerificationError: valid = False
        if not valid or not user or not user.is_active: raise HTTPException(401, 'Email or password is incorrect')
        if hasher.check_needs_rehash(user.password_hash): user.password_hash = hasher.hash(body.password)
        old = request.cookies.get(COOKIE)
        if old: db.execute(delete(Session).where(Session.token_hash == digest(old)))
        user.last_login = time.time()
        issue(db, user, response)
        db.add(AuditEvent(actor_id=user.id, action='LOGIN', subject_id=user.id)); db.commit()
        return public_user(user)

@router.post('/auth/logout')
def logout(request: Request, response: Response, user=Depends(current_user)):
    with DB() as db:
        db.execute(delete(Session).where(Session.token_hash == digest(request.cookies.get(COOKIE, ''))))
        db.add(AuditEvent(actor_id=user.id, action='LOGOUT', subject_id=user.id)); db.commit()
    response.delete_cookie(COOKIE, path='/')
    return {'ok': True}

@router.get('/auth/me')
def me(user=Depends(current_user)): return public_user(user)

@router.post('/auth/refresh')
def refresh(request: Request, response: Response, user=Depends(current_user)):
    with DB() as db:
        db.execute(delete(Session).where(Session.token_hash == digest(request.cookies.get(COOKIE, ''))))
        issue(db, user, response); db.commit()
    return public_user(user)

class Profile(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    avatar_url: str | None = Field(default=None, max_length=2000)
    preferences: dict = Field(default_factory=dict)
    @field_validator('avatar_url')
    @classmethod
    def safe_avatar(cls, v):
        if v and not v.startswith('https://'): raise ValueError('Avatar must use HTTPS')
        return v
    @field_validator('preferences')
    @classmethod
    def safe_preferences(cls, v):
        allowed = {'theme': {'atmosphere', 'night'}, 'notifications': {True, False}, 'start_page': {'Overview', 'Change Impact', 'History'}}
        if any(k not in allowed or not isinstance(value, (str, bool)) or value not in allowed[k] for k, value in v.items()): raise ValueError('Invalid preference')
        return v

@router.patch('/auth/profile')
def profile(body: Profile, user=Depends(current_user)):
    with DB() as db:
        row = db.get(User, user.id)
        row.display_name = body.display_name.strip()
        if not row.display_name: raise HTTPException(422, 'Display name required')
        row.avatar_url = body.avatar_url; row.preferences = body.preferences; row.updated_at = time.time()
        db.add(AuditEvent(actor_id=user.id, action='PROFILE_UPDATED', subject_id=user.id))
        db.commit(); return public_user(row)

class PasswordChange(BaseModel):
    current_password: str = Field(max_length=128)
    password: str = Field(max_length=128)

@router.post('/auth/password')
def change_password(body: PasswordChange, request: Request, response: Response, user=Depends(current_user)):
    limited(request, 'password'); strong(body.password)
    try: hasher.verify(user.password_hash, body.current_password)
    except VerificationError: raise HTTPException(400, 'Current password is incorrect')
    with DB() as db:
        db.get(User, user.id).password_hash = hasher.hash(body.password)
        db.execute(delete(Session).where(Session.user_id == user.id)); issue(db, user, response)
        db.add(AuditEvent(actor_id=user.id, action='PASSWORD_CHANGED', subject_id=user.id)); db.commit()
    return {'ok': True}

class Forgot(BaseModel):
    email: str = Field(max_length=254)

@router.post('/auth/forgot-password')
def forgot(body: Forgot, request: Request):
    limited(request, 'forgot')
    if not os.getenv('SMTP_HOST'): raise HTTPException(503, 'Password recovery email is not configured. Contact your administrator.')
    with DB() as db:
        user = db.scalar(select(User).where(User.email == body.email.strip().lower(), User.is_active == True))
        if user:
            token = secrets.token_urlsafe(48)
            db.execute(delete(PasswordReset).where(PasswordReset.user_id == user.id))
            db.add(PasswordReset(token_hash=digest(token), user_id=user.id, expires_at=time.time()+1800)); db.commit()
            import smtplib
            from email.message import EmailMessage
            message = EmailMessage(); message['Subject'] = 'Reset your Impact Console password'
            message['From'] = os.environ['SMTP_FROM']; message['To'] = user.email
            message.set_content(f"Reset your password within 30 minutes: {os.getenv('APP_URL', 'http://localhost:5173')}/?reset={token}")
            with smtplib.SMTP(os.environ['SMTP_HOST'], int(os.getenv('SMTP_PORT', '587')), timeout=10) as smtp:
                smtp.starttls()
                if os.getenv('SMTP_USER'): smtp.login(os.environ['SMTP_USER'], os.environ['SMTP_PASSWORD'])
                smtp.send_message(message)
    return {'message': 'If the account exists, a reset link will be emailed.'}

class Reset(BaseModel):
    token: str = Field(max_length=200)
    password: str = Field(max_length=128)

@router.post('/auth/reset-password')
def reset(body: Reset, request: Request):
    limited(request, 'reset'); strong(body.password)
    with DB() as db:
        row = db.scalar(select(PasswordReset).where(PasswordReset.token_hash == digest(body.token)).with_for_update())
        if not row or row.expires_at < time.time(): raise HTTPException(400, 'Reset link is invalid or expired')
        account = db.get(User, row.user_id)
        account.password_hash = hasher.hash(body.password)
        db.execute(delete(Session).where(Session.user_id == row.user_id)); db.delete(row)
        db.add(AuditEvent(actor_id=account.id, action='PASSWORD_CHANGED', subject_id=account.id, details={'method': 'reset'})); db.commit()
    return {'ok': True}
