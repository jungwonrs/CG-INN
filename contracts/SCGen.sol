// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract SCGen {
    address public owner;
    uint256 private nonce;

    struct VC {
        bytes32 vcId;
        address adU;
        uint256 tIss;
        uint256 tExp;
        bytes32 hDi;
        bytes sigmaU;
        string status;
    }

    mapping(bytes32 => VC) public vcStore;

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        nonce = 0;
    }

    function issueVC(
        bytes32 _hDi,
        uint256 _tExp,
        bytes calldata _sigmaU
    ) external onlyOwner returns (bytes32) {
        require(block.timestamp < _tExp, "Expired");
        nonce += 1;
        bytes32 vcId = keccak256(abi.encodePacked(owner, _hDi, block.timestamp, nonce));
        vcStore[vcId] = VC({
            vcId: vcId,
            adU: owner,
            tIss: block.timestamp,
            tExp: _tExp,
            hDi: _hDi,
            sigmaU: _sigmaU,
            status: "active"
        });
        return vcId;
    }

    function getVC(bytes32 _vcId) external view onlyOwner returns (VC memory) {
        require(vcStore[_vcId].adU == owner, "Not yours");
        return vcStore[_vcId];
    }

    function revokeVC(bytes32 _vcId) external onlyOwner {
        require(vcStore[_vcId].adU == owner, "Not yours");
        require(keccak256(bytes(vcStore[_vcId].status)) == keccak256(bytes("active")), "Not active");
        vcStore[_vcId].status = "revoked";
    }

    function isValidVC(bytes32 _vcId) external view returns (bool) {
        VC storage vc = vcStore[_vcId];
        return (
            keccak256(bytes(vc.status)) == keccak256(bytes("active")) &&
            block.timestamp < vc.tExp
        );
    }
}
