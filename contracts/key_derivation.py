import hashlib
import hmac
import secrets
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


def sha256_hex(*args: str) -> str:
    return hashlib.sha256("||".join(args).encode()).hexdigest()


def sign(sk: str, msg: str) -> str:
    return hmac.new(sk.encode(), msg.encode(), hashlib.sha256).hexdigest()


def derive_key(ctx: str, sk: str) -> str:
    sig_ctx = sign(sk, ctx)
    pwd = sha256_hex(ctx, sig_ctx)
    h_sk = sha256_hex(sk)
    return format(int(h_sk, 16) ^ int(pwd, 16), '064x')


@dataclass
class VC:
    vc_id: str
    ad_u: str
    t_iss: float
    t_exp: float
    h_di: str
    sigma_u: str
    status: str = "active"


class SCGenSim:
    def __init__(self, deployer: str):
        self.owner = deployer
        self.nonce = 0
        self.store: Dict[str, VC] = {}

    def issue(self, h_di: str, t_exp: float, sigma_u: str, caller: str) -> VC:
        assert caller == self.owner
        import time
        self.nonce += 1
        vc_id = sha256_hex(self.owner, h_di, str(time.time()), str(self.nonce))
        vc = VC(vc_id=vc_id, ad_u=self.owner, t_iss=time.time(),
                t_exp=t_exp, h_di=h_di, sigma_u=sigma_u)
        self.store[vc_id] = vc
        return vc

    def is_valid(self, vc_id: str) -> bool:
        import time
        if vc_id not in self.store:
            return False
        vc = self.store[vc_id]
        return vc.status == "active" and time.time() < vc.t_exp


class SCVerSim:
    def __init__(self, sc_gen: SCGenSim):
        self.sc_gen = sc_gen
        self.ctx_store: Dict[str, str] = {}

    def init_verification(self, vc: VC, sk: str) -> str:
        assert self.sc_gen.is_valid(vc.vc_id)
        h_vc = sha256_hex(vc.vc_id, vc.ad_u, str(vc.t_iss), str(vc.t_exp), vc.h_di, vc.sigma_u)
        ctx = sha256_hex(h_vc, vc.h_di, vc.ad_u)
        self.ctx_store[vc.vc_id] = ctx
        return ctx


class KeyPipeline:
    def __init__(self):
        self.users: Dict[str, dict] = {}

    def register(self, uid: str) -> dict:
        sk = secrets.token_hex(32)
        addr = sha256_hex(uid, sk)[:40]
        info = {'sk': sk, 'addr': addr, 'sc_gen': SCGenSim(addr)}
        self.users[uid] = info
        return info

    def gen_key(self, uid: str, di_hash: str) -> str:
        import time
        u = self.users[uid]
        h_di = sha256_hex(di_hash)
        t_exp = time.time() + 365 * 24 * 3600
        msg = sha256_hex(u['addr'], h_di, str(t_exp))
        sigma = sign(u['sk'], msg)
        vc = u['sc_gen'].issue(h_di, t_exp, sigma, u['addr'])
        actual_msg = sha256_hex(vc.vc_id, vc.ad_u, str(vc.t_iss), str(vc.t_exp), vc.h_di)
        vc.sigma_u = sign(u['sk'], actual_msg)
        sc_ver = SCVerSim(u['sc_gen'])
        ctx = sc_ver.init_verification(vc, u['sk'])
        return derive_key(ctx, u['sk'])
