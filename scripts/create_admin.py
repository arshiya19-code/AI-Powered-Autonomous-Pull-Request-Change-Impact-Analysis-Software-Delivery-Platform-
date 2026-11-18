"""Promote an existing registered account from the trusted server console."""
import argparse
from sqlalchemy import select
from backend.app.database import DB, User, AuditEvent, initialize
if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('email'); args = parser.parse_args()
    initialize()
    with DB() as db:
        user = db.scalar(select(User).where(User.email == args.email.strip().lower()))
        if not user: raise SystemExit('Register the account first.')
        user.role = 'ADMIN'; user.is_active = True
        db.add(AuditEvent(actor_id=user.id, action='ADMIN_BOOTSTRAP', subject_id=user.id)); db.commit()
        print('Administrator access enabled.')
