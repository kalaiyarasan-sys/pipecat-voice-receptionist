"""Clinic scheduling logic used by the voice agent's tool calls.

Kept free of any Pipecat imports so it can be unit-tested and reused
by other channels (WhatsApp, web chat) without the voice pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

SLOT_MINUTES = 30
OPEN_TIME = time(9, 0)
CLOSE_TIME = time(17, 0)
LUNCH_START = time(13, 0)
LUNCH_END = time(14, 0)
BOOKING_WINDOW_DAYS = 14


@dataclass(frozen=True)
class Doctor:
    id: str
    name: str
    specialty: str
    working_days: tuple[int, ...]  # 0 = Monday


DOCTORS: dict[str, Doctor] = {
    "dr-priya": Doctor("dr-priya", "Dr. Priya Raman", "General Physician", (0, 1, 2, 3, 4, 5)),
    "dr-arun": Doctor("dr-arun", "Dr. Arun Kumar", "Pediatrics", (0, 2, 4)),
    "dr-meena": Doctor("dr-meena", "Dr. Meena Sundar", "Dermatology", (1, 3, 5)),
}


@dataclass
class Booking:
    id: str
    patient_name: str
    phone: str
    doctor_id: str
    slot: datetime
    created_at: datetime = field(default_factory=datetime.now)


class BookingError(ValueError):
    """Raised for requests the agent should explain back to the caller."""


def _find_doctor(query: str) -> Doctor:
    q = query.strip().lower().replace("dr.", "").replace("doctor", "").strip()
    for doc in DOCTORS.values():
        if q in (doc.id, doc.name.lower()) or q in doc.name.lower() or q == doc.specialty.lower():
            return doc
    raise BookingError(
        f"No doctor matches '{query}'. Available: "
        + ", ".join(f"{d.name} ({d.specialty})" for d in DOCTORS.values())
    )


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise BookingError(f"Date '{value}' must be in YYYY-MM-DD format.") from exc


def _parse_time(value: str) -> time:
    v = value.strip().upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I %p", "%I:%M%p", "%I%p"):
        try:
            return datetime.strptime(v, fmt).time()
        except ValueError:
            continue
    raise BookingError(f"Time '{value}' is not a valid time, e.g. 10:30 or 3 PM.")


def _normalise_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if len(digits) != 10:
        raise BookingError("Phone number must have 10 digits.")
    return digits


class Clinic:
    def __init__(self, today: date | None = None):
        self._today_override = today
        self._bookings: dict[str, Booking] = {}

    @property
    def today(self) -> date:
        return self._today_override or date.today()

    # ----- queries -------------------------------------------------------

    def list_doctors(self) -> list[dict]:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return [
            {
                "doctor_id": d.id,
                "name": d.name,
                "specialty": d.specialty,
                "days": ", ".join(days[i] for i in d.working_days),
            }
            for d in DOCTORS.values()
        ]

    def _all_slots(self, doctor: Doctor, day: date) -> list[datetime]:
        if day.weekday() not in doctor.working_days:
            return []
        slots, current = [], datetime.combine(day, OPEN_TIME)
        end = datetime.combine(day, CLOSE_TIME)
        while current < end:
            if not (LUNCH_START <= current.time() < LUNCH_END):
                slots.append(current)
            current += timedelta(minutes=SLOT_MINUTES)
        return slots

    def _taken(self, doctor_id: str) -> set[datetime]:
        return {b.slot for b in self._bookings.values() if b.doctor_id == doctor_id}

    def check_availability(self, doctor: str, day: str) -> dict:
        doc = _find_doctor(doctor)
        d = _parse_date(day)
        self._check_window(d)
        if d.weekday() not in doc.working_days:
            return {"doctor": doc.name, "date": d.isoformat(), "available": [],
                    "note": f"{doc.name} does not work on {d.strftime('%A')}."}
        taken = self._taken(doc.id)
        now = datetime.now()
        free = [s for s in self._all_slots(doc, d) if s not in taken and s > now]
        return {
            "doctor": doc.name,
            "date": d.isoformat(),
            # Only a few options: long lists sound bad when spoken aloud.
            "available": [s.strftime("%I:%M %p").lstrip("0") for s in free[:6]],
            "total_free": len(free),
        }

    # ----- actions -------------------------------------------------------

    def book(self, patient_name: str, phone: str, doctor: str, day: str, slot_time: str) -> dict:
        if not patient_name.strip():
            raise BookingError("Patient name is required.")
        doc = _find_doctor(doctor)
        d = _parse_date(day)
        self._check_window(d)
        t = _parse_time(slot_time)
        slot = datetime.combine(d, t)
        if slot not in self._all_slots(doc, d):
            raise BookingError(f"{t.strftime('%I:%M %p')} on {d.isoformat()} is not a valid slot for {doc.name}.")
        if slot <= datetime.now():
            raise BookingError("That time has already passed.")
        if slot in self._taken(doc.id):
            raise BookingError("That slot was just taken. Please choose another time.")
        booking = Booking(
            id=uuid.uuid4().hex[:6].upper(),
            patient_name=patient_name.strip(),
            phone=_normalise_phone(phone),
            doctor_id=doc.id,
            slot=slot,
        )
        self._bookings[booking.id] = booking
        return {
            "booking_id": booking.id,
            "doctor": doc.name,
            "date": d.isoformat(),
            "time": slot.strftime("%I:%M %p").lstrip("0"),
            "patient_name": booking.patient_name,
        }

    def cancel(self, booking_id: str, phone: str) -> dict:
        booking = self._bookings.get(booking_id.strip().upper())
        if not booking or booking.phone != _normalise_phone(phone):
            raise BookingError("No booking found with that ID and phone number.")
        del self._bookings[booking.id]
        return {"cancelled": booking.id}

    def _check_window(self, d: date) -> None:
        if d < self.today:
            raise BookingError("That date is in the past.")
        if d > self.today + timedelta(days=BOOKING_WINDOW_DAYS):
            raise BookingError(f"Bookings open only {BOOKING_WINDOW_DAYS} days ahead.")
