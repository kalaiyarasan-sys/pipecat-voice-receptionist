from datetime import date, timedelta

import pytest

from receptionist.clinic import BookingError, Clinic
from receptionist.prompts import build_system_prompt


def next_weekday(weekday: int) -> date:
    """Next date (at least tomorrow) falling on the given weekday."""
    d = date.today() + timedelta(days=1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d


@pytest.fixture
def clinic():
    return Clinic()


def test_lists_doctors(clinic):
    names = [d["name"] for d in clinic.list_doctors()]
    assert "Dr. Priya Raman" in names


def test_availability_skips_lunch_and_caps_options(clinic):
    day = next_weekday(0).isoformat()  # Monday, Dr. Priya works
    result = clinic.check_availability("Priya", day)
    assert len(result["available"]) == 6
    assert result["total_free"] == 14  # 9-1 and 2-5, 30-min slots


def test_doctor_off_day(clinic):
    result = clinic.check_availability("Arun", next_weekday(1).isoformat())  # Tuesday
    assert result["available"] == []
    assert "does not work" in result["note"]


def test_book_then_slot_is_taken(clinic):
    day = next_weekday(0).isoformat()
    booking = clinic.book("Kumar", "+91 98765 43210", "Priya", day, "10:30 AM")
    assert len(booking["booking_id"]) == 6
    with pytest.raises(BookingError, match="taken"):
        clinic.book("Someone", "9876500000", "Priya", day, "10:30")
    assert clinic.check_availability("Priya", day)["total_free"] == 13


def test_lunch_slot_rejected(clinic):
    with pytest.raises(BookingError, match="not a valid slot"):
        clinic.book("Kumar", "9876543210", "Priya", next_weekday(0).isoformat(), "1 PM")


def test_specialty_lookup(clinic):
    result = clinic.check_availability("pediatrics", next_weekday(0).isoformat())
    assert result["doctor"] == "Dr. Arun Kumar"


@pytest.mark.parametrize("phone", ["12345", "98765432101"])
def test_bad_phone(clinic, phone):
    with pytest.raises(BookingError, match="10 digits"):
        clinic.book("Kumar", phone, "Priya", next_weekday(0).isoformat(), "10:00")


def test_date_outside_window(clinic):
    with pytest.raises(BookingError, match="days ahead"):
        clinic.check_availability("Priya", (date.today() + timedelta(days=30)).isoformat())
    with pytest.raises(BookingError, match="past"):
        clinic.check_availability("Priya", (date.today() - timedelta(days=1)).isoformat())


def test_cancel_needs_matching_phone(clinic):
    day = next_weekday(0).isoformat()
    booking = clinic.book("Kumar", "9876543210", "Priya", day, "11:00")
    with pytest.raises(BookingError):
        clinic.cancel(booking["booking_id"], "9000000000")
    assert clinic.cancel(booking["booking_id"].lower(), "9876543210")["cancelled"] == booking["booking_id"]


def test_prompt_includes_today():
    assert date.today().isoformat() in build_system_prompt()
