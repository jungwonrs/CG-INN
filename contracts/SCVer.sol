// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ISCGen {
    function isValidVC(bytes32 _vcId) external view returns (bool);
}

contract SCVer {
    ISCGen public scGen;
    mapping(bytes32 => bytes32) public ctxStore;

    constructor(address _scGen) {
        scGen = ISCGen(_scGen);
    }

    function initVerification(
        bytes32 _vcId,
        address _adU,
        bytes32 _hVc,
        bytes32 _hDi,
        bytes calldata _sigmaU
    ) external returns (bytes32) {
        require(scGen.isValidVC(_vcId), "VC not valid");
        require(msg.sender == _adU, "Not VC owner");
        require(_verifySig(_hVc, _hDi, _sigmaU, _adU), "Sig failed");

        bytes32 ctx = keccak256(abi.encodePacked(_hVc, _hDi, _adU));
        ctxStore[_vcId] = ctx;
        return ctx;
    }

    function getContext(bytes32 _vcId) external view returns (bytes32) {
        require(ctxStore[_vcId] != bytes32(0), "No ctx");
        return ctxStore[_vcId];
    }

    function _verifySig(
        bytes32 _hVc,
        bytes32 _hDi,
        bytes calldata _sigmaU,
        address _adU
    ) internal pure returns (bool) {
        bytes32 msgHash = keccak256(abi.encodePacked(_hVc, _hDi));
        bytes32 ethHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", msgHash));
        (bytes32 r, bytes32 s, uint8 v) = _splitSig(_sigmaU);
        return ecrecover(ethHash, v, r, s) == _adU;
    }

    function _splitSig(bytes calldata sig) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "Bad sig len");
        r = bytes32(sig[0:32]);
        s = bytes32(sig[32:64]);
        v = uint8(bytes1(sig[64:65]));
    }
}
