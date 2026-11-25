-- ============================================================================
-- CORE SCHEMA (self-contained: runs on a brand-new Supabase project)
--
-- Hospitals, users and roles, patients, dated measurements (vitals + labs),
-- predictions + explanations, and a database-level audit trail.
-- The backend-specific tables (reports, OCR values, evidence, clinical reviews,
-- API audit) are in 20261101000100_backend_gateway.sql, which must run AFTER this file.
--
-- Security principles built in from the start:
--   * Row Level Security (RLS) is ON for every table; there is no "allow all" policy.
--   * Every staff read/write policy checks the caller's ROLE, not just their hospital.
--   * Signup can only create a 'patient' or a (pending) 'doctor'. Privileged roles
--     are assigned later by an administrator, never by signup form data.
--   * A doctor has NO clinical access until an administrator approves them.
-- ============================================================================

-- ---- Types ------------------------------------------------------------------
create type public.user_role as enum
  ('super_admin', 'hospital_admin', 'doctor', 'nurse', 'data_scientist', 'patient');
create type public.severity_level as enum ('Low', 'Moderate', 'High', 'Critical');
create type public.doctor_status as enum ('pending', 'approved', 'rejected');

-- ---- Tables -----------------------------------------------------------------
create table public.hospitals (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    address text,
    region text,
    created_at timestamptz not null default now()
);

-- One row per login (auth.users), created automatically by the signup trigger below.
create table public.profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    full_name text not null default '',
    email text,
    role public.user_role not null default 'patient',
    hospital_id uuid references public.hospitals(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Doctor verification: a doctor can use clinical features only once status = 'approved'.
create table public.doctors (
    user_id uuid primary key references public.profiles(id) on delete cascade,
    license_number text not null,
    specialization text,
    status public.doctor_status not null default 'pending',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- The clinical record of a patient (different from the patient's LOGIN in profiles).
-- user_id links the two; it is NULL until an administrator links them.
create table public.patients (
    id uuid primary key default gen_random_uuid(),
    hospital_id uuid not null references public.hospitals(id) on delete cascade,
    mrn text not null unique,                              -- medical record number
    demographics jsonb not null default '{}'::jsonb,       -- first_name, last_name, gender, age, ...
    family_history jsonb not null default '[]'::jsonb,
    user_id uuid unique references public.profiles(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Dated measurements (time series)
create table public.vitals (
    id uuid primary key default gen_random_uuid(),
    patient_id uuid not null references public.patients(id) on delete cascade,
    type text not null,                                    -- e.g. 'blood_pressure_systolic'
    value numeric not null,
    unit text not null,
    recorded_at timestamptz not null,
    created_at timestamptz not null default now()
);
create index vitals_patient_time_idx on public.vitals (patient_id, recorded_at desc);

create table public.lab_results (
    id uuid primary key default gen_random_uuid(),
    patient_id uuid not null references public.patients(id) on delete cascade,
    test_name text not null,                               -- e.g. 'HbA1c'
    value numeric not null,
    unit text not null,
    reference_range text,
    recorded_at timestamptz not null,
    created_at timestamptz not null default now()
);
create index lab_results_patient_time_idx on public.lab_results (patient_id, recorded_at desc);

-- Model output. (The backend adds input_features, source_measurement_ids and warnings.)
create table public.predictions (
    id uuid primary key default gen_random_uuid(),
    patient_id uuid not null references public.patients(id) on delete cascade,
    disease text not null,                                 -- 'Diabetes' | 'Heart Disease' | 'CKD'
    probability numeric not null check (probability >= 0 and probability <= 1),
    confidence numeric not null check (confidence >= 0 and confidence <= 1),
    severity public.severity_level not null,
    model_version text not null,
    created_at timestamptz not null default now()
);
create index predictions_patient_time_idx on public.predictions (patient_id, created_at desc);

-- Exactly one explanation per prediction.
create table public.explanations (
    id uuid primary key default gen_random_uuid(),
    prediction_id uuid not null unique references public.predictions(id) on delete cascade,
    shap_payload jsonb not null default '{}'::jsonb,
    clinical_explanation text not null,
    created_at timestamptz not null default now()
);

-- Database-level audit trail of data CHANGES, written by triggers only.
create table public.audit_logs (
    id uuid primary key default gen_random_uuid(),
    actor_id uuid references public.profiles(id) on delete set null,
    action text not null check (action in ('SELECT', 'INSERT', 'UPDATE', 'DELETE')),
    table_name text not null,
    row_id uuid not null,
    old_values jsonb,
    new_values jsonb,
    created_at timestamptz not null default now()
);

-- ---- Helper functions (security definer = run past RLS, so policies cannot recurse) -------
create or replace function public.update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- The caller's effective role. A doctor who is not (yet) approved counts as a plain
-- 'patient', so RLS gives an unapproved doctor no clinical access at all.
create or replace function public.get_my_role()
returns public.user_role as $$
  select case
           when p.role = 'doctor' and not exists (
             select 1 from public.doctors d where d.user_id = p.id and d.status = 'approved'
           ) then 'patient'::public.user_role
           else p.role
         end
  from public.profiles p
  where p.id = auth.uid();
$$ language sql security definer stable set search_path = public;

create or replace function public.get_my_hospital_id()
returns uuid as $$
  select hospital_id from public.profiles where id = auth.uid();
$$ language sql security definer stable set search_path = public;

-- ---- Triggers ------------------------------------------------------------------------------
create trigger update_profiles_updated_at before update on public.profiles
  for each row execute function public.update_updated_at_column();
create trigger update_doctors_updated_at before update on public.doctors
  for each row execute function public.update_updated_at_column();
create trigger update_patients_updated_at before update on public.patients
  for each row execute function public.update_updated_at_column();

-- Signup: create the profile (and a pending doctor record). The signup form data is chosen by
-- the user, so it may only select 'patient' or 'doctor'. Anything else becomes 'patient'.
create or replace function public.handle_new_user()
returns trigger as $$
declare
  v_role public.user_role;
begin
  v_role := case when new.raw_user_meta_data->>'role' = 'doctor'
                 then 'doctor'::public.user_role else 'patient'::public.user_role end;

  insert into public.profiles (id, full_name, email, role, hospital_id)
  values (new.id,
          coalesce(new.raw_user_meta_data->>'full_name', ''),
          new.email,
          v_role,
          nullif(new.raw_user_meta_data->>'hospital_id', '')::uuid);

  if v_role = 'doctor' and nullif(new.raw_user_meta_data->>'license_number', '') is not null then
    insert into public.doctors (user_id, license_number, specialization)
    values (new.id, new.raw_user_meta_data->>'license_number', new.raw_user_meta_data->>'specialization');
  end if;
  return new;
end;
$$ language plpgsql security definer set search_path = public;

create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

-- Users may edit their own profile, but NOT their role or hospital. Only administrators can.
-- (auth.uid() is null for migrations, the SQL editor and the backend's service role: trusted.)
create or replace function public.protect_profile_privileges()
returns trigger as $$
begin
  if auth.uid() is null then
    return new;
  end if;
  if new.role is distinct from old.role or new.hospital_id is distinct from old.hospital_id then
    if public.get_my_role() = 'super_admin' then
      return new;
    elsif public.get_my_role() = 'hospital_admin'
          and old.hospital_id = public.get_my_hospital_id()
          and new.hospital_id is not distinct from old.hospital_id
          and new.role <> 'super_admin' then
      return new;
    end if;
    raise exception 'Only administrators can change a role or hospital.';
  end if;
  return new;
end;
$$ language plpgsql security definer set search_path = public;

create trigger protect_profile_privileges_trigger before update on public.profiles
  for each row execute function public.protect_profile_privileges();

-- Records every change to sensitive tables (who, what, before and after).
create or replace function public.log_phi_access()
returns trigger as $$
declare
    act text;
    r_id uuid;
    old_val jsonb := null;
    new_val jsonb := null;
begin
    if TG_OP = 'INSERT' then
        act := 'INSERT'; r_id := new.id; new_val := to_jsonb(new);
    elsif TG_OP = 'UPDATE' then
        act := 'UPDATE'; r_id := new.id; old_val := to_jsonb(old); new_val := to_jsonb(new);
    else
        act := 'DELETE'; r_id := old.id; old_val := to_jsonb(old);
    end if;
    insert into public.audit_logs (actor_id, action, table_name, row_id, old_values, new_values)
    values (auth.uid(), act, TG_TABLE_NAME, r_id, old_val, new_val);
    return coalesce(new, old);
end;
$$ language plpgsql security definer set search_path = public;

create trigger audit_patients after insert or update or delete on public.patients
  for each row execute function public.log_phi_access();
create trigger audit_predictions after insert or update or delete on public.predictions
  for each row execute function public.log_phi_access();
create trigger audit_lab_results after insert or update or delete on public.lab_results
  for each row execute function public.log_phi_access();
create trigger audit_vitals after insert or update or delete on public.vitals
  for each row execute function public.log_phi_access();

-- ---- Row Level Security --------------------------------------------------------------------
alter table public.hospitals enable row level security;
alter table public.profiles enable row level security;
alter table public.doctors enable row level security;
alter table public.patients enable row level security;
alter table public.vitals enable row level security;
alter table public.lab_results enable row level security;
alter table public.predictions enable row level security;
alter table public.explanations enable row level security;
alter table public.audit_logs enable row level security;

-- hospitals: names are public (the signup form needs a list); only a super admin edits them.
create policy "Hospitals: anyone can read" on public.hospitals for select using (true);
create policy "Hospitals: super_admin manage" on public.hospitals
  for all using (public.get_my_role() = 'super_admin');

-- profiles
create policy "Profiles: read own" on public.profiles for select using (id = auth.uid());
create policy "Profiles: admins read" on public.profiles for select using (
  public.get_my_role() = 'super_admin'
  or (public.get_my_role() = 'hospital_admin' and hospital_id = public.get_my_hospital_id())
);
create policy "Profiles: update own" on public.profiles
  for update using (id = auth.uid()) with check (id = auth.uid());     -- role/hospital guarded by trigger
create policy "Profiles: admins update" on public.profiles for update using (
  public.get_my_role() = 'super_admin'
  or (public.get_my_role() = 'hospital_admin' and hospital_id = public.get_my_hospital_id())
);

-- doctors: a doctor sees their own verification status; administrators approve or reject.
create policy "Doctors: read own" on public.doctors for select using (user_id = auth.uid());
create policy "Doctors: admins manage" on public.doctors for all using (
  public.get_my_role() = 'super_admin'
  or (public.get_my_role() = 'hospital_admin' and exists (
        select 1 from public.profiles p
        where p.id = user_id and p.hospital_id = public.get_my_hospital_id()))
);

-- patients (patient-side read-own access is added in the backend gateway migration)
create policy "Patients: super_admin full access" on public.patients
  for all using (public.get_my_role() = 'super_admin');
create policy "Patients: hospital_admin manage same hospital" on public.patients
  for all using (public.get_my_role() = 'hospital_admin' and hospital_id = public.get_my_hospital_id());
create policy "Patients: doctor read same hospital" on public.patients
  for select using (public.get_my_role() = 'doctor' and hospital_id = public.get_my_hospital_id());
create policy "Patients: doctor create same hospital" on public.patients
  for insert with check (public.get_my_role() = 'doctor' and hospital_id = public.get_my_hospital_id());
create policy "Patients: doctor update same hospital" on public.patients
  for update using (public.get_my_role() = 'doctor' and hospital_id = public.get_my_hospital_id());
create policy "Patients: nurse read same hospital" on public.patients
  for select using (public.get_my_role() = 'nurse' and hospital_id = public.get_my_hospital_id());

-- vitals and lab_results: same shape. Doctors and hospital admins write; nurses read.
create policy "Vitals: super_admin full access" on public.vitals
  for all using (public.get_my_role() = 'super_admin');
create policy "Vitals: doctor/hospital_admin manage same hospital" on public.vitals for all using (
  public.get_my_role() in ('doctor', 'hospital_admin')
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);
create policy "Vitals: nurse read same hospital" on public.vitals for select using (
  public.get_my_role() = 'nurse'
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);

create policy "LabResults: super_admin full access" on public.lab_results
  for all using (public.get_my_role() = 'super_admin');
create policy "LabResults: doctor/hospital_admin manage same hospital" on public.lab_results for all using (
  public.get_my_role() in ('doctor', 'hospital_admin')
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);
create policy "LabResults: nurse read same hospital" on public.lab_results for select using (
  public.get_my_role() = 'nurse'
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);

-- predictions: doctors create them; clinical staff of the same hospital read them.
create policy "Predictions: super_admin full access" on public.predictions
  for all using (public.get_my_role() = 'super_admin');
create policy "Predictions: data_scientist read all" on public.predictions
  for select using (public.get_my_role() = 'data_scientist');
create policy "Predictions: staff read same hospital" on public.predictions for select using (
  public.get_my_role() in ('doctor', 'nurse', 'hospital_admin')
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);
create policy "Predictions: doctor insert same hospital" on public.predictions for insert with check (
  public.get_my_role() = 'doctor'
  and exists (select 1 from public.patients p where p.id = patient_id and p.hospital_id = public.get_my_hospital_id())
);

create policy "Explanations: super_admin full access" on public.explanations
  for all using (public.get_my_role() = 'super_admin');
create policy "Explanations: data_scientist read all" on public.explanations
  for select using (public.get_my_role() = 'data_scientist');
create policy "Explanations: staff read same hospital" on public.explanations for select using (
  public.get_my_role() in ('doctor', 'nurse', 'hospital_admin')
  and exists (
    select 1 from public.predictions pr join public.patients p on p.id = pr.patient_id
    where pr.id = prediction_id and p.hospital_id = public.get_my_hospital_id())
);
create policy "Explanations: doctor insert same hospital" on public.explanations for insert with check (
  public.get_my_role() = 'doctor'
  and exists (
    select 1 from public.predictions pr join public.patients p on p.id = pr.patient_id
    where pr.id = prediction_id and p.hospital_id = public.get_my_hospital_id())
);

-- audit_logs: read-only for administrators (a hospital admin sees only their own hospital's actors).
-- There is no INSERT/UPDATE/DELETE policy: only the triggers above can write.
create policy "AuditLogs: super_admin read" on public.audit_logs
  for select using (public.get_my_role() = 'super_admin');
create policy "AuditLogs: hospital_admin read own hospital" on public.audit_logs for select using (
  public.get_my_role() = 'hospital_admin'
  and exists (select 1 from public.profiles pr where pr.id = actor_id and pr.hospital_id = public.get_my_hospital_id())
);
